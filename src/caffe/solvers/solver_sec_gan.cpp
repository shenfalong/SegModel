#include <cstdio>
#include <stdio.h>
#include <string>
#include <vector>
#include "caffe/util/format.hpp"
#include "caffe/solvers/solver_sec_gan.hpp"

#include "caffe/util/io.hpp"


namespace caffe {

template <typename Dtype>
SolverSecGAN<Dtype>::SolverSecGAN(const SolverParameter& param)
{
  //---------------- initial train net -----------------------
  param_ = param;
  NetParameter g_net_param;
  ReadProtoFromTextFile(param_.g_net(), &g_net_param);

	Caffe::set_bn_state(param_.bn_state());
	Caffe::set_drop_state(param_.drop_state());
	
	//------------------- generative part------------------------
  vector<shared_ptr<Blob<Dtype> > > net_intput_blobs; 
  net_intput_blobs.clear();
  vector<std::string> net_intput_blob_names; 
  net_intput_blob_names.clear();
  g_net_.reset(new Net<Dtype>(g_net_param, net_intput_blobs, net_intput_blob_names));
  g_net_->set_NetOptimizer(param_.g_net_opt());
  //------------------------------------------------------
  	
  	
	//-------------------
  int num_output_blobs = g_net_->output_blobs().size();
  net_intput_blobs.clear(); 
  net_intput_blob_names.clear();
  net_intput_blobs.resize(num_output_blobs);
  net_intput_blob_names.resize(num_output_blobs);
  CHECK_EQ(g_net_->output_blobs().size(),g_net_->output_blob_indices().size());
  for (int i = 0;i < num_output_blobs; i++)
  {
		net_intput_blobs[i].reset(new Blob<Dtype>());
		net_intput_blobs[i]->ReshapeLike(*g_net_->output_blobs()[i]);
		net_intput_blob_names[i] = g_net_->blob_names()[g_net_->output_blob_indices()[i]]; 
	}
	//-------------------
				

	//-------------------- discriminative part--------------------------
  NetParameter d_net_param;
  ReadProtoFromTextFile(param_.d_net(), &d_net_param);
  d_net_.reset(new Net<Dtype>(d_net_param, net_intput_blobs, net_intput_blob_names));
  d_net_->set_NetOptimizer(param_.d_net_opt());
	//------------------------------------------------------
	
	
	//------------------- GRADIENT PENALTY---------------------------
	NetParameter d_interp_net_param;
	ReadProtoFromTextFile(param_.d_interp_net(), &d_interp_net_param);
	d_interp_net_.reset(new Net<Dtype>(d_interp_net_param, net_intput_blobs, net_intput_blob_names));
  d_interp_net_->set_NetOptimizer(param_.d_net_opt());
	this->share_weight(d_net_, d_interp_net_);
	//------------------------------------------------------
	
  this->iter_ = 0;
}

template <typename Dtype>
void SolverSecGAN<Dtype>::Solve(const char* all_state_file) 
{
  LOG(INFO) << "Solving Generative Adversial Networks";


  if (all_state_file)
  {
   	LOG(INFO) << "Restoring previous solver status from " << all_state_file;
    Restore(all_state_file);
  }

  start_iter_ = this->iter_;
	std::vector<Dtype> g_loss;
	std::vector<Dtype> d_loss;
	g_loss.clear();
	d_loss.clear();
  while (this->iter_ < param_.max_iter())
  {
  	CUDA_SYNCHRONIZE;
    g_net_->ClearParamDiffs();
    d_net_->ClearParamDiffs();    
    Dtype cur_loss = 0;
//-----------------------------------------------		
		Caffe::set_gan_type("train_dnet");	
		g_net_->BcastData();		
		for(int i_critic = 0;i_critic<5;i_critic++)
		{
			g_net_->Forward();
				
			for (int ig = 0;ig < g_net_->output_blobs().size(); ig++)
			{
				//LOG(INFO)<<g_net_->blob_names()[g_net_->output_blob_indices()[ig]];
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[ig%NGPUS]));
				d_net_->input_blobs()[ig]->ReshapeLike(*g_net_->output_blobs()[ig]);
				caffe_copy(g_net_->output_blobs()[ig]->count(),g_net_->output_blobs()[ig]->gpu_data(),d_net_->input_blobs()[ig]->mutable_gpu_data());
			}
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	
			d_net_->BcastData();
			cur_loss = d_net_->Forward();
			d_net_->Backward();

		
			//-------------------gradient penalty-----------------
			g_net_->Forward();
			
			for (int ig = 0;ig < g_net_->output_blobs().size(); ig++)
			{
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[ig%NGPUS]));
				d_interp_net_->input_blobs()[ig]->ReshapeLike(*g_net_->output_blobs()[ig]);
				caffe_copy(g_net_->output_blobs()[ig]->count(),g_net_->output_blobs()[ig]->gpu_data(),d_interp_net_->input_blobs()[ig]->mutable_gpu_data());
			}
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
			
			d_interp_net_->Forward();
			Caffe::set_frozen_param(true);
			d_interp_net_->Backward();
			Caffe::set_frozen_param(false);
			Caffe::set_second_pass(true);
			d_interp_net_->SecForward();
			Caffe::set_second_pass(false);
			//-----------------------------------------------	
			
			d_net_->ReduceDiff();
			d_net_->Update(this->iter_, param_.max_iter(), this->iter_ % param_.display() == 0 && i_critic == 0);
			d_net_->ClearParamDiffs();  				
		}
		d_loss.push_back(cur_loss);
//----------------------------------------------- 	
		Caffe::set_gan_type("train_gnet");		
	
		g_net_->Forward();
	
		for (int ig = 0;ig < g_net_->output_blobs().size(); ig++)
		{	
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[ig%NGPUS]));
			d_net_->input_blobs()[ig]->ReshapeLike(*g_net_->output_blobs()[ig]);
			caffe_copy(g_net_->output_blobs()[ig]->count(),g_net_->output_blobs()[ig]->gpu_data(),d_net_->input_blobs()[ig]->mutable_gpu_data());
		}
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		
	
		d_net_->BcastData();
		cur_loss = d_net_->Forward();
		Caffe::set_frozen_param(true);
		d_net_->Backward();
		Caffe::set_frozen_param(false);
		//-----unecessary-------
	
		for (int ig = 0;ig < g_net_->output_blobs().size(); ig++)
		{
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[ig%NGPUS]));
			caffe_copy(g_net_->output_blobs()[ig]->count(),d_net_->input_blobs()[ig]->gpu_diff(),g_net_->output_blobs()[ig]->mutable_gpu_diff());
		}
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));

		g_net_->Backward();
		g_loss.push_back(cur_loss);
	
		
		g_net_->ReduceDiff();
		g_net_->Update(this->iter_, param_.max_iter(), false);
		g_net_->ClearParamDiffs();     
//-----------------------------------------------
    ++this->iter_;
    
    if (this->iter_ % param_.display() == 0)
    {
    	dispaly_loss(g_loss, d_loss);
    	g_loss.clear(); d_loss.clear();
    }
    if (param_.snapshot() && this->iter_ % param_.snapshot() == 0 )
    	Snapshot();
  }
  //at the end of the training, accumulate the batch norm statisticals 
  if (param_.accumulate_batch_norm())
  {
//-----------------------------------------------
  	LOG(INFO)<<"====accumulate statisticals of samples for batch norm=====";
  	Caffe::number_collect_sample = 0;
  	for (int i = 0; i < param_.accumulate_max_iter(); ++i)
		{	
			g_net_->Forward();
			d_net_->Forward();
			Caffe::number_collect_sample ++;
		}	
		Caffe::number_collect_sample = -1;
  } 
	//Finally, save the model 
	//Snapshot();
	
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void SolverSecGAN<Dtype>::dispaly_loss(std::vector<Dtype> g_loss,std::vector<Dtype> d_loss) 
{
	//LOG(INFO) << "Iteration " << this->iter_ << " ------------";
	Dtype g_sum = 0;
	for (int i=0;i<g_loss.size();i++)
		g_sum += g_loss[i];
	Dtype d_sum = 0;
	for (int i=0;i<d_loss.size();i++)
		d_sum += d_loss[i];
	LOG(INFO) << " g_loss ="<< g_sum/g_loss.size()<<", d_loss ="<< d_sum/d_loss.size();
}

///------------------------------------------------ proto <->  memory------------------------------------------------------
template <typename Dtype>
void SolverSecGAN<Dtype>::Snapshot() 
{
//------------------------------
	NetParameter g_net_param;
  g_net_->ToProto(&g_net_param);
  string g_model_filename = param_.snapshot_prefix() + "_GNet_iter_" +format_int(this->iter_) + ".caffemodel";
  WriteProtoToBinaryFile(g_net_param, g_model_filename);
  
  NetParameter d_net_param;
  d_net_->ToProto(&d_net_param);
  string d_model_filename = param_.snapshot_prefix() + "_DNet_iter_" +format_int(this->iter_) + ".caffemodel";
	WriteProtoToBinaryFile(d_net_param, d_model_filename);
	
	LOG(INFO) << "Snapshotting to binary proto file " << g_model_filename<< " and "<<d_model_filename;
//-----------------------------

//-----------------------------	

	SolverState g_state;
	g_state.set_learned_net(g_model_filename);
	g_net_->StateToProto(g_state.mutable_net_state());
	string g_snapshot_filename = param_.snapshot_prefix() + "_GNet_iter_"+ format_int(this->iter_)+ ".solverstate";
	WriteProtoToBinaryFile(g_state, g_snapshot_filename.c_str());
	
	SolverState d_state;
	d_state.set_learned_net(d_model_filename);
	d_net_->StateToProto(d_state.mutable_net_state());
	string d_snapshot_filename = param_.snapshot_prefix() + "_DNet_iter_"+ format_int(this->iter_)+ ".solverstate";
	WriteProtoToBinaryFile(d_state, d_snapshot_filename.c_str());
	
	LOG(INFO) << "Snapshotting to binary proto file " << g_snapshot_filename<<" and "<<d_snapshot_filename;
//-----------------------------

//-----------------------------
	SolverState all_state;
	all_state.set_iter(this->iter_);
	all_state.set_g_state_file(g_snapshot_filename);
	all_state.set_d_state_file(d_snapshot_filename);
	string all_snapshot_filename = param_.snapshot_prefix() + "_all_iter_"+ format_int(this->iter_)+ ".solverstate";
	WriteProtoToBinaryFile(all_state, all_snapshot_filename.c_str());
	
	LOG(INFO) << "Snapshotting to binary proto file " << all_snapshot_filename;
}


template <typename Dtype>
void SolverSecGAN<Dtype>::Restore(const char* all_state_file) 
{
	SolverState all_state;
	ReadProtoFromBinaryFile(all_state_file, &all_state);
	this->iter_ = all_state.iter();
	const char * g_state_file = all_state.g_state_file().c_str();
	const char * d_state_file = all_state.d_state_file().c_str();
	

	SolverState g_state;
	ReadProtoFromBinaryFile(g_state_file, &g_state);
	NetParameter g_net_param;
  ReadProtoFromBinaryFile(g_state.learned_net().c_str(), &g_net_param);
  g_net_->CopyTrainedLayersFrom(g_net_param);
  g_net_->RestoreState(g_state.net_state());
  
  SolverState d_state;
	ReadProtoFromBinaryFile(d_state_file, &d_state);
	NetParameter d_net_param;
  ReadProtoFromBinaryFile(d_state.learned_net().c_str(), &d_net_param);
  d_net_->CopyTrainedLayersFrom(d_net_param);
  d_net_->RestoreState(d_state.net_state());
}

INSTANTIATE_CLASS(SolverSecGAN);

}  // namespace caffe
