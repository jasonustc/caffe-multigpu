//added by qing li 2014-12-27
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <time.h>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "boost/scoped_ptr.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace caffe;
using boost::scoped_ptr;
using std::string;

DEFINE_bool(gray, false, "when this option is on, treat images as grayscale ones.");
DEFINE_bool(shuffle, false, "Randomly shuffle the order of images and their labels.");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_string(enc, "", "the coding format of image");
DEFINE_double(start_prop, 0, "The start proportion of the data set");
DEFINE_double(end_prop, 1, "The end proportion of the data set");
DEFINE_int32(rand_crop_num, 10, "The number of random crops");



void convert_image_data(const string folder_name, const string& img_idx_file_name, const string &db_name,
	std::string& enc, const int rand_N ,const bool gray, const bool shuffle,const int resize_width,
	const int resize_height, const double start_prop, const double end_prop){
	std::ifstream infile(img_idx_file_name);
	std::vector<std::pair<std::string, int>> lines;
	std::string imgname;
	int label;

	while (infile >> imgname >> label){
		lines.push_back(std::make_pair(imgname, label));
	}

	//shuffle order of images
	if (shuffle){
		LOG(INFO) << "Shuffling data.";
		std::random_shuffle(lines.begin(), lines.end());
		size_t pos = img_idx_file_name.rfind('.');
		string ext = pos == img_idx_file_name.npos ? img_idx_file_name : img_idx_file_name.substr(pos);
		string file_name = img_idx_file_name.substr(0, pos);
		const std::string shuffle_file = file_name + "_shuffled" + ext;
		std::ofstream outfile(shuffle_file);
		CHECK(outfile.is_open());
		for (size_t l = 0; l < lines.size(); l++){
			outfile << lines[l].first << "\t" << lines[l].second << "\n";
		}
		outfile.close();
		LOG(INFO) << "Save shuffled index of images into file: " << shuffle_file;
	}
	LOG(INFO) << "A total of " << lines.size() << " images.";

	//Create new DB
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(options, db_name, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_name <<
		" ,maybe it already exists.";

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	//save image data into leveldb
	string img_file_path;
	bool stat;
	cv::Mat ori_img;
	int count = 0;
	clock_t time1, time2;
	time1 = clock();
	Datum pair_datum;
	//2 images, 3 channels
	pair_datum.set_channels( 2 * 3 );
	pair_datum.set_width(resize_width);
	pair_datum.set_height(resize_height);
	pair_datum.set_encoded(enc.size() > 0);
	string pair_img_buffer;
	Datum resize_datum;
	Datum crop_datum;
	cv::Mat rand_crop_img;
	string img_buffer;
	string crop_img_buffer;
	const int start_id = lines.size() * start_prop;
	const int end_id = lines.size() * end_prop;


	//build train set
	int num_pairs = 0;
	for (int line_id = std::max(start_id,0); line_id < std::min(end_id, int(lines.size())); line_id++){
		pair_datum.set_label(lines[line_id].second);
		img_file_path = folder_name + lines[line_id].first;
		stat = ReadImageToDatum(img_file_path, label, resize_height, resize_width,
			true, enc, &resize_datum);
		if (stat == false){
			LOG(ERROR)<< "Failed to read image '" << img_file_path << "'.";
			continue;
		}
		img_buffer.clear();
		img_buffer = resize_datum.data();
		ori_img = ReadImageToCVMat(img_file_path, true);
		CHECK(ori_img.depth() == CV_8U) << "Image data type must be unsigned byte.";
		if(!ori_img.data){
			LOG(ERROR) << "Failed to read mat data of image '" << img_file_path << "'.";
			continue;
		}
		for (int randi = 0; randi < rand_N; randi++){
			rand_crop_img = RandomCropCVMat(ori_img, resize_width, resize_height);
			//empty mat
			if (!rand_crop_img.data){
				continue;
			}
			CVMatToDatum(rand_crop_img, &crop_datum);
			crop_img_buffer.clear();
			crop_img_buffer = crop_datum.data();
			pair_img_buffer.clear();
	  		pair_img_buffer.append(img_buffer);
			pair_img_buffer.append(crop_img_buffer);
			pair_datum.clear_data();
			pair_datum.clear_float_data();
			pair_datum.set_data(pair_img_buffer);
			//sequential
			string out;
			pair_datum.SerializeToString(&out);
			int length = sprintf_s(key_cstr, kMaxKeyLength, "%08d_%03d_%s", line_id, randi,
				lines[line_id].first.c_str());
//			//Put in db
			leveldb::Status s = db->Put(leveldb::WriteOptions(), std::string(key_cstr, length), out);
			num_pairs++;
		} //for (int randi = 0; randi < rand_N; randi++)
		if (++count % 100 == 0){
			time2 = clock();
			float diff_time((float)time2 - (float)time1);
			diff_time /= CLOCKS_PER_SEC;
			LOG(INFO) << "Processed " << count << " train images in "<<diff_time<<" s";
			LOG(INFO) << "Generated " << num_pairs << " pairs";
		}
	}//for (int line_id = std::max(start_id,0); line_id < std::min(end_id, int(lines.size())); line_id++)
	LOG(INFO) << "\nProcessed " << num_pairs << " pairs";
	delete db;
}


int main(int argc, char** argv)
{
	if (argc < 4)
	{
		gflags::SetUsageMessage("Convert a set of images to leveldb\n"
			"format used as input for caffe.\n"
			"usage:\n"
			"    EXE [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
		return 1;
	}
	else {
		google::InitGoogleLogging(argv[0]);
		google::SetStderrLogging(0);
		gflags::ParseCommandLineFlags(&argc, &argv, true);
		const int resize_width = std::max(0, FLAGS_resize_width);
		const int resize_height = std::max(0, FLAGS_resize_height);
		const bool shuffle = FLAGS_shuffle;
		const bool gray_scale = FLAGS_gray;
		string enc = FLAGS_enc;
		const double start_prop = FLAGS_start_prop;
		const double end_prop = FLAGS_end_prop;
		const int rand_crop_num = FLAGS_rand_crop_num;
		LOG(INFO) << " root_folder: " << argv[1] << " list_file: " << argv[2] << " db_name: " << argv[3]
			<< " encoding: " << enc << " rand_crop_num: " << rand_crop_num << " shuffled: " << shuffle
			<< " resize_width: " << resize_width << " resize_height: " << resize_height << " start_prop: "
			<< start_prop << " end_prop: " << end_prop;
//		convert_dataset(argv[1], argv[2], argv[3], resize_width, resize_height, shuffle);
		convert_image_data(argv[1], argv[2], argv[3], enc, rand_crop_num, gray_scale,
			shuffle, resize_width, resize_height, start_prop, end_prop);
	}
    return 0;
}
