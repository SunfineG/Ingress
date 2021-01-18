
#ifndef EXAMPLES_ANALYTICAL_APPS_TORNADO_H_
#define EXAMPLES_ANALYTICAL_APPS_TORNADO_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "examples/analytical_apps/flags.h"
#include "examples/analytical_apps/timer.h"
#include "examples/analytical_apps/tornado/pagerank.h"
#include "examples/analytical_apps/tornado/php.h"

namespace grape {
namespace tornado {

void Init() {
  if (!FLAGS_out_prefix.empty()) {
    if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
      mkdir(FLAGS_out_prefix.c_str(), 0777);
    }
  }

  if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  //  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const CommSpec& comm_spec, const std::string& efile_base,
                    const std::string& efile_updated, const std::string& vfile,
                    const std::string& out_prefix,
                    const ParallelEngineSpec& spec, Args... args) {
  std::shared_ptr<FRAG_T> fragment;
  std::shared_ptr<typename APP_T::worker_t> worker;

  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  worker = APP_T::CreateWorker(app);

  std::vector<std::string> efiles;

  efiles.push_back(efile_base);
  if (!efile_updated.empty()) {
    efiles.push_back(efile_updated);
  }
  int query_id = 0;

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

  for (size_t i = 0; i < efiles.size(); i++) {
    auto& efile = efiles[i];
    if (efile.empty()) {
      continue;
    }
    timer_next("load graph " + efile);
    LOG(INFO) << "Loading fragment";
    LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

    graph_spec.set_directed(FLAGS_directed);
    graph_spec.set_rebalance(false, 0);

    SetSerialize(comm_spec, FLAGS_serialization_prefix, efile, vfile,
                 graph_spec);

    fragment = LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
    worker->Init(comm_spec, spec);
    worker->SetFragment(fragment);
    timer_next("run algorithm, Query: " + std::to_string(query_id++));
    worker->Query(std::forward<Args>(args)...);
    if (i == efiles.size() - 1) {
      timer_end();
    }
    if (i == efiles.size() - 1) {
      if (!out_prefix.empty()) {
        std::ofstream ostream;
        std::string output_path =
            grape::GetResultFilename(out_prefix, fragment->fid());
        ostream.open(output_path);
        worker->Output(ostream);
        ostream.close();
        VLOG(1) << "Worker-" << comm_spec.worker_id()
                << " finished: " << output_path;
      }
    }
    worker->Finalize();
    // release graph
    fragment.reset();
  }
}

}  // namespace tornado
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_TORNADO_H_
