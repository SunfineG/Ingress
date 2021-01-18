
#ifndef GRAPE_WORKER_INGRESS_SYNC_TRAVERSAL_WORKER_H_
#define GRAPE_WORKER_INGRESS_SYNC_TRAVERSAL_WORKER_H_

#include <grape/fragment/loader.h>

#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/ingress_sync_message_manager.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/parallel/parallel_message_manager.h"
#include "timer.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class IterateKernel;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */

template <typename APP_T>
class IngressSyncTraversalWorker : public ParallelEngine {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "IngressSyncTraversalWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = ParallelMessageManager;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename APP_T::vid_t;

  IngressSyncTraversalWorker(std::shared_ptr<APP_T> app,
                             std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());
    messages_.InitChannels(thread_num());
    communicator_.InitCommunicator(comm_spec.comm());

    InitParallelEngine(pe_spec);
    LOG(INFO) << "Thread num: " << getWorkers();
  }

  /**
   * 根据依赖重置
   *
   */
  void ResetByDedend(const std::unordered_set<oid_t>& changed_vertices) {
    std::vector<vid_t> local_gid;
    std::vector<vid_t> local_parent;
    std::vector<std::vector<vid_t>> global_gid_list;
    std::vector<std::vector<vid_t>> global_parent_list;

    auto iv = graph_->InnerVertices();

    // u->v, v's parent is u
    for (auto v : iv) {
      auto parent_gid = app_->value_parent_gid(v);

      local_gid.push_back(graph_->Vertex2Gid(v));
      local_parent.push_back(parent_gid);
    }

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "GlobalAllGatherv";
    }
    GlobalAllGatherv(local_gid, global_gid_list, comm_spec_.comm(),
                     comm_spec_.worker_num());
    GlobalAllGatherv(local_parent, global_parent_list, comm_spec_.comm(),
                     comm_spec_.worker_num());

    // Construct dependency tree, dp_tree[v] contains children
    std::unordered_map<vid_t, std::vector<vid_t>> dp_tree;

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Constructing dp-tree";
    }
    for (size_t i = 0; i < global_gid_list.size(); i++) {
      auto& global_gid = global_gid_list[i];
      auto& global_parent = global_parent_list[i];

      for (size_t j = 0; j < global_gid.size(); j++) {
        auto v = global_gid[j];
        auto p_v = global_parent[j];

        dp_tree[p_v].push_back(v);
      }
    }

    // 重置
    std::unordered_set<vid_t> gid_set;  // 去重，存储需要重置的id
    std::queue<vid_t> gid_queue;        // 重置队列

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Init Q";
    }
    for (auto dst_oid : changed_vertices) {
      vid_t dst_gid;
      CHECK(graph_->Oid2Gid(dst_oid, dst_gid));
      if (gid_set.find(dst_gid) == gid_set.end()) {
        gid_set.insert(dst_gid);
        gid_queue.push(dst_gid);
      }
    }

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Traversing";
    }

    // level traverse dp tree to get all vertices should be reset
    while (!gid_queue.empty()) {
      auto gid = gid_queue.front();

      gid_queue.pop();

      for (auto child_gid : dp_tree[gid]) {  // 将依赖于u的重置
        if (gid_set.find(child_gid) == gid_set.end()) {
          gid_set.insert(child_gid);
          gid_queue.push(child_gid);
        }
      }
    }

    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    LOG(INFO) << "Resetting... size: " << gid_set.size();
    for (auto gid : gid_set) {
      vertex_t v;
      // reset inner vertices
      if (graph_->IsInnerGid(gid) && graph_->InnerVertexGid2Vertex(gid, v)) {
        values[v] = app_->default_v();
        // init c, app_->next_modified_ is useless
        app_->init_c(*graph_, v, deltas[v], app_->next_modified_);
      }
    }
    MPI_Barrier(comm_spec_.comm());
  }

  /**
   * 用于重新加载图，并完成图变换前后值的校正
   *
   */
  void reLoadGrape() {
    std::string change_efile = FLAGS_efile_update;  // 边改变的文件
    std::string new_vfile = FLAGS_vfile;            // 新顶点文件
    std::string new_efile = FLAGS_efile_updated;    // 新边文件

    if (change_efile.empty() || new_efile.empty()) {
      return;
    }

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Loading change file: " << change_efile << ", " << new_vfile
                << ", " << new_efile;
    }
    auto changed_vertices = LoadChageVertex<fragment_t>(change_efile);
    // 重置
    ResetByDedend(changed_vertices);

    // 加载新图的结构
    LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
    graph_spec.set_directed(FLAGS_directed);
    graph_spec.set_rebalance(false, 0);
    SetSerialize(comm_spec_, FLAGS_serialization_prefix, new_efile, new_vfile,
                 graph_spec);

    auto inner_vertices = graph_->InnerVertices();
    VertexArray<value_t, vid_t> values;
    VertexArray<value_t, vid_t> deltas;

    values.Init(inner_vertices);
    deltas.Init(inner_vertices);

    // backup app data
    for (auto v : inner_vertices) {
      values[v] = app_->values_[v];
      deltas[v] = app_->deltas_[v];
    }

    // 利用新图文件加载新图
    graph_.reset();

    graph_ =
        LoadGraph<fragment_t, SegmentedPartitioner<typename fragment_t::oid_t>>(
            new_efile, new_vfile, comm_spec_, graph_spec);

    CHECK_EQ(graph_->InnerVertices().size(), inner_vertices.size());

    app_->Init(comm_spec_, *graph_, true, true);

    auto outer_vertices = graph_->OuterVertices();
    auto vertices = graph_->Vertices();

    // copy to new graph
    for (auto v : inner_vertices) {
      app_->values_[v] = values[v];
      app_->deltas_[v] = deltas[v];
    }

    // 回收
    {
      MPI_Barrier(comm_spec_.comm());
      int step = 1;

      auto& values = app_->values_;
      auto& deltas = app_->deltas_;

      app_->curr_modified_.Clear();
      app_->next_modified_.Clear();

      // 发送消息: 发送要求是每个点都向其出度点无条件发送当前值。
      messages_.StartARound();
      for (auto u : inner_vertices) {
        auto value = values[u];
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->g_function(*graph_, u, app_->default_v(), value, oes,
                         app_->next_modified_);
      }

      // send local delta to remote
      for (auto v : outer_vertices) {
        if (app_->next_modified_.Exist(v)) {
          auto& delta_to_send = deltas[v];

          if (delta_to_send != app_->default_v()) {
            auto p_gid = app_->delta_parent_gid(v);
            std::pair<vid_t, value_t> msg(p_gid, delta_to_send);

            messages_.SyncStateOnOuterVertex(*graph_, v, msg);
            delta_to_send = app_->default_v();
          }
        }
      }

      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      LOG(INFO) << app_->next_modified_.Count();
      // default_work,同步一轮
      messages_.FinishARound();
      MPI_Barrier(comm_spec_.comm());
    }
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());

    // allocate dependency arrays
    app_->Init(comm_spec_, *graph_, true, true);
    int step = 1;
    bool batch_stage = true;

    if (FLAGS_debug) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i) {
        sleep(1);
      }
    }

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    double exec_time = 0;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    while (true) {
      exec_time -= GetCurrentTime();

      messages_.StartARound();
      auto& channels = messages_.Channels();

      app_->next_modified_.ParallelClear(thread_num());

      {
        vertex_t v;
        // 消息是： u -> v
        messages_.ParallelProcess<fragment_t, std::pair<vid_t, value_t>>(
            thread_num(), *graph_,
            [this](int tid, vertex_t v, std::pair<vid_t, value_t> msg) {
              value_t received_delta = msg.second;
              //将delta加到对应的顶点上delta上面
              if (app_->accumulate(app_->deltas_[v], received_delta)) {
                auto p_gid = msg.first;
                app_->add_delta_dependency(p_gid, v);  // v的delta依赖于u
                app_->curr_modified_.Insert(v);
              }
            });
      }

      // Traverse outgoing neighbors
      if (FLAGS_cilk) {
        ForEachCilk(app_->curr_modified_, inner_vertices,
                    [this, &values, &deltas](int tid, vertex_t u) {
                      auto& value = values[u];
                      auto delta = atomic_exch(deltas[u], app_->default_v());
                      auto last_value = value;

                      auto oes = graph_->GetOutgoingAdjList(u);

                      if (app_->accumulate(value, delta)) {
                        app_->g_function(*graph_, u, last_value, delta, oes,
                                         app_->next_modified_);

                        // 说明u的value值被delta更新了，来源与delta_depend[u]
                        app_->mark_value_dependency(u);
                      }
                    });
      } else {
        ForEachSimple(app_->curr_modified_, inner_vertices,
                      [this, &values, &deltas](int tid, vertex_t u) {
                        auto& value = values[u];
                        auto delta = atomic_exch(deltas[u], app_->default_v());
                        auto last_value = value;
                        auto oes = graph_->GetOutgoingAdjList(u);

                        if (app_->accumulate(value, delta)) {
                          app_->g_function(*graph_, u, last_value, delta, oes,
                                           app_->next_modified_);

                          // 说明u的value值被delta更新了，来源与delta_depend[u]
                          app_->mark_value_dependency(u);
                        }
                      });
      }

      // send local delta to remote
      ForEach(app_->next_modified_, outer_vertices,
              [&channels, &deltas, this](int tid, vertex_t v) {
                auto& delta_to_send = deltas[v];
                //减少无用消息的发送
                if (delta_to_send != app_->default_v()) {
                  auto p_gid = app_->delta_parent_gid(v);
                  std::pair<vid_t, value_t> msg(p_gid, delta_to_send);

                  channels[tid].SyncStateOnOuterVertex(*graph_, v, msg);
                  delta_to_send = app_->default_v();
                }
              });

      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      // default_work,同步一轮
      messages_.FinishARound();

      exec_time += GetCurrentTime();

      bool terminate = termCheck(app_->next_modified_);

      if (terminate) {
        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "Batch time: " << exec_time << " sec";
          }
          exec_time = 0;
          step = 1;

          if (!FLAGS_efile_updated.empty() && !FLAGS_efile_update.empty()) {
            reLoadGrape();  // 重新加载图
            inner_vertices = graph_->InnerVertices();
            outer_vertices = graph_->OuterVertices();
            values = app_->values_;
            deltas = app_->deltas_;
          } else {
            LOG(ERROR) << "Missing efile_update or efile_updated";
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "Inc time: " << exec_time << " sec";
          }
          break;
        }
      }

      ++step;
      app_->next_modified_.Swap(app_->curr_modified_);
    }
    MPI_Barrier(comm_spec_.comm());
  }

  void Output(std::ostream& os) {
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;

    for (auto v : inner_vertices) {
      os << graph_->GetId(v) << " " << values[v] << std::endl;
    }
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck(DenseVertexSet<vid_t>& active_set) {
    size_t global_active_count;
    auto local_active_count = active_set.ParallelCount(thread_num());

    communicator_.template Sum(local_active_count, global_active_count);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      VLOG(1) << "Global Active count: " << global_active_count;
    }

    return global_active_count == 0;
  }

  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t>& graph_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
};

}  // namespace grape

#endif