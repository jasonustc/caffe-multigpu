#include <vector>
#include <utility>

#include "caffe/common.hpp"
#include "deep_aesth.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_string(net_file, "imagenet_val.prototxt",
	"net model file for feature extraction");
DEFINE_string(model_file, "bvlc_reference_caffenet.caffemodel",
	"trained parameter model");
DEFINE_bool(sqrt, false,
	"Optional; if we need to do root square on extracted features.");
DEFINE_bool(l2_norm, false,
	"Optional; if we need to do L2 normalization on extracted features.");

int main(int argc, char** argv){
	if (argc < 3){
		gflags::SetUsageMessage("Usage: extract_deep_aesth_feat.exe file_path[image_path, index_file, or folder_path] "
			" blob_name \n"
			" please notice that index_file should be one line an image and with an .txt extension\n");
		gflags::ShowUsageWithFlagsRestrict(*argv, "extract_deep_aesth_feat");
		return -1;
	}
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_logtostderr = true;
	CHECK(check_file_type(FLAGS_net_file) == 2);
	CHECK(check_file_type(FLAGS_model_file) == 2);
	DeepAesth<float> deep_aesth(FLAGS_net_file, FLAGS_model_file);
	string file_path = argv[1];
	string blob_name = argv[2];
	int file_type = check_file_type(file_path);
	if (file_type == 0 || file_type == 1){
		LOG(FATAL) << "not a regular file or not exist";
	}
	else if (file_type == 2){
		// index file
		if (get_ext(file_path) == ".txt"){
			deep_aesth.GetFeatFromIndexFile(file_path, blob_name);
		}
		else{
			// image path
			deep_aesth.GetFeatFromImage(file_path, blob_name);
		}
	}
	else {
		// folder
		deep_aesth.GetFeatFromFolder(file_path, blob_name);
	}
	return 0;
}