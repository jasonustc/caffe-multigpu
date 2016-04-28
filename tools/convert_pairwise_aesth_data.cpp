#include<fstream>
#include<string>
#include<time.h>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb\db.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe\util\math_functions.hpp"
#include "boost\scoped_ptr.hpp"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace caffe;
using boost::scoped_ptr;
using std::string;
using std::pair;

DEFINE_bool(gray, false, "when this option is on, treat image as grayscale ones.");
DEFINE_bool(shuffle, false, "Randomly shuffle the order of images and their labels.");
DEFINE_int32(resize_width, 0, "width images are resized to, if set to 0, don't resize");
DEFINE_int32(resize_height, 0, "height images are resized to, if set to 0, don't resize");
DEFINE_string(enc, "", "the coding format of image");
DEFINE_bool(split_valid, false, "if we need to split the data set into two part.");
DEFINE_double(train_prop, 0.8, "The proportion of training set");

bool is_good(const string& image){
	int ind = image.rfind(".");
	char label = image.c_str()[ind];
	return label == 'G' || label == 'g';
}

void get_image_list(const string& index_file, bool shuffle, 
	std::vector<std::pair<std::string, std::string>>& lines){
	lines.clear();
	ifstream in_file(index_file.c_str());
	CHECK(in_file.is_open()) << "Can not open index image file: " << index_file;
	string good_image;
	string bad_image;
	while (in_file >> good_image >> bad_image){
//		CHECK(is_good(good_image))<<"first image must be a good image: " << good_image;
//		CHECK(!is_good(bad_image))<<"second image must be a bad image: " << bad_image;
		lines.push_back(std::make_pair(good_image, bad_image));
	}
	if (shuffle){
		LOG(INFO) << "Shuffling image data.";
		std::random_shuffle(lines.begin(), lines.end());
		size_t pos = index_file.rfind('.');
		string ext = pos == index_file.npos ? index_file : index_file.substr(pos);
		string file_name = index_file.substr(0, pos);
		const std::string shuffle_file = file_name + "_shuffled" + ext;
		std::ofstream outfile(shuffle_file);
		CHECK(outfile.is_open()) << "can not open shuffling file: " << shuffle_file;
		for (size_t l = 0; l < lines.size(); l++){
			outfile << lines[l].first << "\t" << lines[l].second << "\n";
		}
		outfile.close();
		LOG(INFO) << "Save shuffled index of images into file: " << shuffle_file;
	}
	LOG(INFO) << "A total of " << lines.size() << " images.";
}

void create_db(const string& index_file, const string& db_name,
	std::string& enc, const bool gray, const bool shuffle, const int resize_width,
	const int resize_height, const int start_id, const int end_id, 
	std::vector<std::pair<string, string>>& lines){

	//create new db
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(options, db_name, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_name <<
		", maybe it already exists.";

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	//save image data into leveldb
	string good_image_path;
	string bad_image_path;
	int count = 0;
	clock_t time1, time2;
	time1 = clock();
	Datum pair_datum;
	//2 images,each image with 3 channels
	/*
	 *datum: channels, width, height, encoded, data(), float_data()
	 *export: datum.SerializeToString(&out)
	 */
	pair_datum.set_channels(2 * 3);
	pair_datum.set_width(resize_width);
	pair_datum.set_height(resize_height);
	pair_datum.set_encoded(enc.size() > 0);
	string pair_img_buffer;
	Datum resize_datum_good;
	Datum resize_datum_bad;
	string good_img_buffer;
	string bad_img_buffer;

	bool stat;

	//build training dataset
	int num_pairs = 0;
	const int label = 1;
	for (int line_id = start_id; line_id < end_id; line_id++){
		//always set to 1, because good image is always in front of bad images
		pair_datum.set_label(1);
		good_image_path = lines[line_id].first;
		bad_image_path = lines[line_id].second;
		stat = ReadImageToDatum(good_image_path, label, resize_height, resize_width,
			true, enc, &resize_datum_good);
//		LOG(INFO) << resize_datum_good.encoded();
		if (stat){
			stat = ReadImageToDatum(bad_image_path, label, resize_height, resize_width,
				true, enc, &resize_datum_bad);
		}
		else{
			LOG(ERROR) << "Failed to read image " << good_image_path <<
				", skip read bad image too. ";
		}
		if (!stat){
			continue;
		}
		good_img_buffer.clear();
		good_img_buffer = resize_datum_good.data();
		bad_img_buffer.clear();
		bad_img_buffer = resize_datum_bad.data();
		pair_img_buffer.clear();
		pair_img_buffer.append(good_img_buffer);
		pair_img_buffer.append(bad_img_buffer);
		pair_datum.clear_data();
		pair_datum.clear_float_data();
		pair_datum.set_data(pair_img_buffer);
		//sequential 
		string out;
		pair_datum.SerializeToString(&out);
		int length = sprintf_s(key_cstr, kMaxKeyLength, "%08d_%s_%s", line_id, 
			good_image_path.c_str(), bad_image_path.c_str());
		//put into db
		leveldb::Status s = db->Put(leveldb::WriteOptions(), std::string(key_cstr, length), out);
		num_pairs++;
		if (++count % 100 == 0){
			time2 = clock();
			float diff_time((float)time2 - (float)time1);
			diff_time /= CLOCKS_PER_SEC;
			LOG(INFO) << "Processed " << count << " training images in " << diff_time << " s.";
			LOG(INFO) << "Generated " << num_pairs << " pairs.";
		}
	}// for (int line_id = 0; line_id < train_end; line_id++)
	LOG(INFO) << "Processed " << num_pairs << " pairs";
	delete db;
}

void convert_image_data(const string& index_file, const string& db_name,
	std::string& enc, const bool gray, const bool shuffle, const int resize_width,
	const int resize_height, const bool split_valid = false, const float train_prop = 1.){
	//read index image paths
	std::vector<std::pair<string, string>> lines;
	get_image_list(index_file, shuffle, lines);
	//db proportion
	const int train_end = split_valid == true ? int(train_prop * lines.size()) : lines.size();
	const int valid_start = train_end == lines.size() ? lines.size() : train_end;
	const int valid_end = lines.size();
	//create db
	if (train_end == lines.size()){
		create_db(index_file, db_name, enc, gray, shuffle, resize_width,
			resize_height, 0, train_end, lines);
	}
	else{
		string train_db_name = db_name + "_train";
		create_db(index_file, train_db_name, enc, gray, shuffle, resize_width,
			resize_height, 0, train_end, lines);
		string valid_db_name = db_name + "_valid";
		create_db(index_file, valid_db_name, enc, gray, shuffle, resize_width,
			resize_height, valid_start, valid_end, lines);
	}
}

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	if (argc < 3){
		gflags::SetUsageMessage("Convert a set of pair images to leveldb\n"
			"format used as input for caffe.\n"
			"usage: \n"
			" EXE [FLAGS] LISTFILE DB_NAME\n");
//		LOG(INFO) << gflags::CommandlineFlagsIntoString();
		gflags::ShowUsageWithFlags(argv[0]);
		return 1;
	}

	google::SetStderrLogging(0);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	const int resize_width = std::max(0, FLAGS_resize_width);
	const int resize_height = std::max(0, FLAGS_resize_height);
	const bool shuffle = FLAGS_shuffle;
	const bool gray_scale = FLAGS_gray;
	string enc = FLAGS_enc;
	const bool split_valid = FLAGS_split_valid;
	const double train_prop = FLAGS_train_prop;
	LOG(INFO) << "list_file: " << argv[1] << " db_name: " << argv[2] <<
		" encoding: " << enc << " shuffle: " << shuffle << " resize_width: " <<
		resize_width << " resize_height: " << resize_height << " splid valid set: " <<
		split_valid << " train set proportion: " << train_prop;
	convert_image_data(argv[1], argv[2], enc, gray_scale, shuffle, 
		resize_width, resize_height, split_valid, train_prop);
	return 0;
}