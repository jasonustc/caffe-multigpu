#include <vector>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/feat_extractor.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_string(net_file, "imagenet_val.prototxt",
	"net model file for feature extraction");
DEFINE_string(model_file, "bvlc_reference_caffenet.caffemodel",
	"trained parameter model");
DEFINE_string(feat_file, "",
	"optional, if provided, feats of all the images will be put into this file\n" \
	"if not, feats will be saved independently into xxx.jpg.feat");
DEFINE_string(mode, "CPU",
	"optional, use CPU or GPU");
DEFINE_bool(sqrt, false,
	"Optional; if we need to do root square on extracted features.");
DEFINE_bool(l2_norm, false,
	"Optional; if we need to do L2 normalization on extracted features.");

using namespace caffe;

int main(int argc, char** argv){
	if (argc < 3){
		gflags::SetUsageMessage("Usage: extract_feat.exe file_path[image_path, index_file, or folder_path] "
			" blob_name \n"
			" please notice that index_file should be one line an image and with an .txt extension\n");
		gflags::ShowUsageWithFlagsRestrict(*argv, "extract_feat");
		return -1;
	}
	::google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_logtostderr = true;
	CHECK(check_file_type(FLAGS_net_file) == 2);
	CHECK(check_file_type(FLAGS_model_file) == 2);
	if (!FLAGS_feat_file.empty()){
		CHECK(!check_file_type(FLAGS_feat_file)) << "please delete " 
			<< FLAGS_feat_file << " first";
	}
	FeatExtractor<float>* feat_extractor;
	Caffe::Brew mode = (FLAGS_mode == "GPU") ? Caffe::GPU : Caffe::CPU;
	if (FLAGS_feat_file.size()){
		feat_extractor = new FeatExtractor<float>(FLAGS_net_file, 
			FLAGS_model_file, FLAGS_feat_file, mode);
	}
	else{
		feat_extractor = new FeatExtractor<float>(FLAGS_net_file, FLAGS_model_file, mode);
	}
	string file_path = argv[1];
	string blob_name = argv[2];
	int file_type = check_file_type(file_path);
	if (file_type == 0 || file_type == 1){
		LOG(FATAL) << "not a regular file or not exist";
	}
	else if (file_type == 2){
		// index file
		if (get_ext(file_path) == ".txt"){
			feat_extractor->GetFeatFromIndexFile(file_path, blob_name);
		}
		else{
			// image path
			feat_extractor->GetFeatFromImage(file_path, blob_name);
		}
	}
	else {
		// folder
		feat_extractor->GetFeatFromFolder(file_path, blob_name);
	}
	return 0;
}