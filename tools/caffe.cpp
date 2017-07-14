#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"


using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
              "Optional; run in GPU mode on given device IDs separated by ','."
              "Use '-gpu all' to run on all available GPUs. The effective training "
              "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
              "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
              "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
              "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
              "Optional; the pretrained weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(g_weights, "",
              "Optional; the pretrained g_weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(d_weights, "",
              "Optional; the pretrained d_weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");              
DEFINE_int32(iterations, 50,
             "The number of iterations to run.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
  namespace { \
  class __Registerer_##func { \
  public: /* NOLINT */ \
  __Registerer_##func() { \
  g_brew_map[#func] = &func; \
  } \
  }; \
  __Registerer_##func g_registerer_##func; \
  }

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus)
{
  if (FLAGS_gpu.size())
  {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i)
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
  }
  else
    LOG(FATAL)<<"no GPU is available";

}


//---------------------------------------------------------------- train ------------------------------------------------------------------
int train()
{
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size()) << "Give a snapshot to resume training or weights to finetune " "but not both.";




  vector<int> gpus;
  gpus.clear();
  get_gpus(&gpus);


  Caffe::GPUs.clear();
  for (int i=0;i<gpus.size();i++)
    Caffe::GPUs.push_back(gpus[i]);
  if (gpus.size() == 0)
    LOG(FATAL) << "Please choose at least one GPU";
  else
  {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i)
      s << (i ? ", " : "") << gpus[i];

    LOG(INFO) << "Using GPUs " << s.str();
  }
	
	caffe::Caffe::Get();

  caffe::SolverParameter solver_param;
  ReadProtoFromTextFile(FLAGS_solver, &solver_param);
  if (solver_param.solver_type() == "CNN")
  {
		shared_ptr<caffe::SolverCNN<float> > solver_cnn(new caffe::SolverCNN<float>(solver_param));
		if (FLAGS_snapshot.size())
		{
		  LOG(INFO) << "Resuming from " << FLAGS_snapshot;
		  solver_cnn->Restore(FLAGS_snapshot.c_str());
		}
		else if (FLAGS_weights.size())
		{
		  LOG(INFO) << "Finetuning from " << FLAGS_weights;
		  
		  vector<string> strings;
		  boost::split(strings, FLAGS_weights, boost::is_any_of(","));
		  for (int i = 0; i < strings.size(); ++i)
		  {
				caffe::NetParameter net_param;
				ReadProtoFromBinaryFile(strings[i], &net_param);  
				solver_cnn->net()->CopyTrainedLayersFrom(net_param);
			}
		}
	  LOG(INFO) << "Starting Optimization";
		solver_cnn->Solve();
		LOG(INFO) << "Optimization Done.";
	}
	else if (solver_param.solver_type() == "GAN")
	{
		shared_ptr<caffe::SolverGAN<float> > solver_gan(new caffe::SolverGAN<float>(solver_param));
		if (FLAGS_snapshot.size())
		{
		  LOG(INFO) << "Resuming from " << FLAGS_snapshot;
		  solver_gan->Restore(FLAGS_snapshot.c_str());
		}
		else if (FLAGS_g_weights.size() || FLAGS_d_weights.size())
		{
			if (FLAGS_g_weights.size())
			{
				LOG(INFO) << "Finetuning from " << FLAGS_g_weights;
				caffe::NetParameter g_net_param;
				ReadProtoFromBinaryFile(FLAGS_g_weights, &g_net_param);
				solver_gan->g_net()->CopyTrainedLayersFrom(g_net_param);
			}
			if (FLAGS_d_weights.size())
			{
				LOG(INFO) << "Finetuning from " << FLAGS_d_weights;
				caffe::NetParameter d_net_param;
				ReadProtoFromBinaryFile(FLAGS_d_weights, &d_net_param);
				solver_gan->d_net()->CopyTrainedLayersFrom(d_net_param);
		  }		  
		}

		LOG(INFO) << "Starting Optimization";
		solver_gan->Solve();
		LOG(INFO) << "Optimization Done.";
	}
	else if (solver_param.solver_type() == "SecGAN")
	{
		shared_ptr<caffe::SolverSecGAN<float> > solver_gan(new caffe::SolverSecGAN<float>(solver_param));
		if (FLAGS_snapshot.size())
		{
		  LOG(INFO) << "Resuming from " << FLAGS_snapshot;
		  solver_gan->Restore(FLAGS_snapshot.c_str());
		}
		else if (FLAGS_g_weights.size() || FLAGS_d_weights.size())
		{
			if (FLAGS_g_weights.size())
			{
				LOG(INFO) << "Finetuning from " << FLAGS_g_weights;
				caffe::NetParameter g_net_param;
				ReadProtoFromBinaryFile(FLAGS_g_weights, &g_net_param);
				solver_gan->g_net()->CopyTrainedLayersFrom(g_net_param);
			}
			if (FLAGS_d_weights.size())
			{
				LOG(INFO) << "Finetuning from " << FLAGS_d_weights;
				caffe::NetParameter d_net_param;
				ReadProtoFromBinaryFile(FLAGS_d_weights, &d_net_param);
				solver_gan->d_net()->CopyTrainedLayersFrom(d_net_param);
		  }		  
		}

		LOG(INFO) << "Starting Optimization";
		solver_gan->Solve();
		LOG(INFO) << "Optimization Done.";
	}
	else
		LOG(FATAL)<<"wrong solver type";


  return 0;
}
RegisterBrewFunction(train);
//----------------------------------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------  time --------------------------------------------------------
int time()
{
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

#ifndef CPU_ONLY
  vector<int> gpus;
  get_gpus(&gpus);

  Caffe::GPUs.clear();
  for (int i=0;i<gpus.size();i++)
    Caffe::GPUs.push_back(gpus[i]);


  if (gpus.size() != 0)
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#endif


  caffe::NetParameter net_param;
  ReadProtoFromTextFile(FLAGS_model, &net_param);
  vector<shared_ptr<Blob<float> > > net_intput_blobs; 
	net_intput_blobs.clear();
	vector<std::string> net_intput_blob_names; 
	net_intput_blob_names.clear();
  Net<float> caffe_net(net_param,net_intput_blobs,net_intput_blob_names);



  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j)
  {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i)
    {
      timer.Start();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i)
    {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: " << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i)
  {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername << "\tforward: " << forward_time_per_layer[i] / 1000 / FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  << "\tbackward: " << backward_time_per_layer[i] / 1000 / FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 / FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 / FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() / FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";

  return 0;
}
RegisterBrewFunction(time);
//----------------------------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  FLAGS_alsologtostderr = 1;
  gflags::SetUsageMessage("command line brew\n"
                          "usage: caffe <command> <args>\n\n"
                          "commands:\n"
                          "  train           train or finetune a model\n"
                          "  test            score a model\n"
                          "  device_query    show GPU diagnostic information\n"
                          "  time            benchmark model execution time");
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2)
    return GetBrewFunction(caffe::string(argv[1]))();
  else
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
}
