#include<fstream>
#include<string>
#include<time.h>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe\util\math_functions.hpp"
#include "boost\scoped_ptr.hpp"
#include "opencv2\core\core.hpp"
#include "caffe/util/db.hpp"

using namespace std;
using namespace caffe;
using boost::scoped_ptr;
using std::string;
using std::pair;

DEFINE_bool(gray, false, "when this option is on, treat image as grayscale ones.");
DEFINE_bool(shuffle, false, "Randomly shuffle the order of images and their labels.");
DEFINE_int32(resize_width, 256, "width images are resized to, if set to 0, don't resize");
DEFINE_int32(resize_height, 256, "height images are resized to, if set to 0, don't resize");
DEFINE_string(enc, "", "the coding format of image");
DEFINE_bool(split_valid, false, "if we need to split the data set into two part.");
DEFINE_double(train_prop, 0.8, "The proportion of training set");
DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");
DEFINE_bool(check_size, false, "When this option is on, check that all the datum have the same size.");
DEFINE_bool(is_feat_file, false, "If the path of input data is feat file(in float format).");

bool is_good(const string& image){
	int ind = image.rfind(".");
	char label = image.c_str()[ind];
	return label == 'G' || label == 'g';
}

void get_image_list(const string& index_file, bool shuffle, 
	std::vector<std::pair<std::string, int>>& lines){
	lines.clear();
	ifstream in_file(index_file.c_str());
	CHECK(in_file.is_open()) << "Can not open index image file: " << index_file;
	string image;
	int label;
	while (in_file >> image >> label){
//		CHECK(is_good(good_image))<<"first image must be a good image: " << good_image;
//		CHECK(!is_good(bad_image))<<"second image must be a bad image: " << bad_image;
		lines.push_back(std::make_pair(image, label));
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
	std::vector<std::pair<string, int>>& lines, const bool is_feat_file = false){

	//create new db in leveldb or hmdb format
	//build deep learning feature and handcraft feature into different db.
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(db_name, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	//save image data into db for deep learning
	string image_path;
	int count = 0;
	clock_t time1, time2;
	time1 = clock();
	Datum img_datum;
	/*
	 *datum: channels, width, height, encoded, data(), float_data()
	 *export: datum.SerializeToString(&out)
	 */
	const int channels = is_feat_file == 1 ? 1 : 3;
	img_datum.set_channels(channels);
	img_datum.set_height(resize_height);
	img_datum.set_width(resize_width);
	img_datum.set_encoded(enc.size() > 0);
	string img_buffer;

	bool stat;

	//build training dataset
	int num_imgs = 0;
	int data_size = 0;
	bool data_size_initialized = false;
	int label;
	float feat;
	for (int line_id = start_id; line_id < end_id; line_id++){
		img_datum.clear_data();
		img_datum.clear_float_data();
		img_datum.clear_label();
		//always set to 1, because good image is always in front of bad images
		image_path = lines[line_id].first;
		label = lines[line_id].second;
		if (!is_feat_file){
			//read data from jpg file
			stat = ReadImageToDatum(image_path, label, resize_height, resize_width,
				true, enc, &img_datum);
			if (!stat){
				LOG(FATAL) << "Can not load image " << image_path;
			}
		}
		else{
			//read data from feat file
			ifstream inFeat(image_path);
			while (inFeat >> feat){
				img_datum.add_float_data(feat);
			}
			if (!inFeat.is_open()){
				LOG(FATAL) << "Can not load image " << image_path;
			}
			img_datum.set_label(label);
			inFeat.close();
		}
		//check size
		if (FLAGS_check_size){
			if (!data_size_initialized){
				data_size = img_datum.channels() * img_datum.height() * img_datum.width();
				data_size_initialized = true;
			}
			else{
				const int size = img_datum.data().size() > 0 ? 
					img_datum.data().size() : img_datum.float_data_size();
				CHECK_EQ(size, data_size) << "Incorrect field data size " << size;
			}
		}
		//sequential 
		string out;
		img_datum.SerializeToString(&out);
		int length = sprintf_s(key_cstr, kMaxKeyLength, "%08d_%s", line_id, image_path);
		//put into db
		txn->Put(string(key_cstr, length), out);
		if (++count % 1000 == 0){
			//commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " images.";
		}
	}// for (int line_id = 0; line_id < train_end; line_id++)
	if (count % 1000 != 0){
		txn->Commit();
		LOG(INFO) << "Processed " << count << " images.";
	}
}

void convert_image_data(const string& index_file, const string& db_name,
	std::string& enc, const bool gray, const bool shuffle, const int resize_width,
	const int resize_height, const bool split_valid = false, const float train_prop = 1.){
	//read index image paths
	std::vector<std::pair<string, int>> lines;
	get_image_list(index_file, shuffle, lines);
	//db proportion
	const int train_end = split_valid == true ? int(train_prop * lines.size()) : lines.size();
	const int valid_start = train_end == lines.size() ? lines.size() : train_end;
	const int valid_end = lines.size();
	//create db
	if (train_end == lines.size()){
		create_db(index_file, db_name, enc, gray, shuffle, resize_width,
			resize_height, 0, train_end, lines, FLAGS_is_feat_file);
	}
	else{
		string train_db_name = db_name + "_train";
		create_db(index_file, train_db_name, enc, gray, shuffle, resize_width,
			resize_height, 0, train_end, lines, FLAGS_is_feat_file);
		string valid_db_name = db_name + "_valid";
		create_db(index_file, valid_db_name, enc, gray, shuffle, resize_width,
			resize_height, valid_start, valid_end, lines, FLAGS_is_feat_file);
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