//added by qing li 2014-12-27
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

using namespace std;
using namespace caffe;


void read_feature(std::pair<int, int> feature_pair, std::vector<std::string> video2feature_list, int dim, caffe::Datum& datum) {

	std::ifstream feature1(video2feature_list[feature_pair.first], std::ios::in);
	std::ifstream feature2(video2feature_list[feature_pair.second], std::ios::in);
	float value;
	for (int i = 0; i<dim; i++)
	{
		feature1 >> value;
		datum.add_float_data(value);
	}

	for (int i = dim; i<2 * dim; i++)
	{
		feature2 >> value;
		datum.add_float_data(value);
	}
}

void convert_dataset(const char* video2feature_list_filename, const char* pair_list_filename,
	const char* db_filename, int feature_dim, int shuffle) {
	// Open files
	std::ifstream video2feature_list_in(video2feature_list_filename, std::ios::in);
	std::ifstream pair_list_in(pair_list_filename, std::ios::in);
	CHECK(video2feature_list_in) << "Unable to open file " << video2feature_list_filename;
	CHECK(pair_list_in) << "Unable to open file " << pair_list_filename;

	//read metadata
	map<int, vector<string>> video2feature_list;
	int video;
	string feature;
	while (video2feature_list_in >> video >> feature)
	{
		if (video2feature_list.find(video) == video2feature_list.end())
			video2feature_list.insert(pair<int, vector<string>>(video, vector<string>()));
		video2feature_list[video].push_back(feature);
	}

	int index1;
	int index2;

	std::vector<std::pair<int, int>> pair_list;
	while (pair_list_in >> index1 >> index2)
	{
		pair_list.push_back(std::make_pair(index1, index2));
		//LOG(INFO)<<"INDEX:"<<index1<<'\t'<<index2;
	}

	video2feature_list_in.close();
	pair_list_in.close();


	if (shuffle == 1)
	{
		LOG(INFO) << "Shuffling data";
		std::random_shuffle(pair_list.begin(), pair_list.end());
	}
	int num_pairs = pair_list.size();

	LOG(INFO) << "A total of " << num_pairs << " pairs";

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	options.write_buffer_size = 256 * 1024 * 1024;
	options.max_open_files = 2000;
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_filename
		<< ". Is it already existing?";

	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	const int kMaxKeyLength = 256;
	char key[kMaxKeyLength];
	std::string out;

	for (int pairid = 0; pairid < num_pairs; ++pairid) {
		Datum merge_datum;
		vector<string> feature_list1 = video2feature_list[pair_list[pairid].first];
		vector<string> feature_list2 = video2feature_list[pair_list[pairid].second];
		CHECK(feature_list1.size() == feature_list2.size()) << "the pair have different num of features" << '\n';
		merge_datum.set_channels(feature_list1.size() * feature_dim * 2);
		merge_datum.set_width(1);
		merge_datum.set_height(1);
		for (int i = 0; i < feature_list1.size(); i++)
		{
			string feature = feature_list1[i]; 
			std::ifstream feature_in(feature, std::ios::in);
			float value;
			for (int i = 0; i<feature_dim; i++)
			{
				feature_in >> value;
				merge_datum.add_float_data(value);
			}
			feature_in.close();
		}
		for (int i = 0; i < feature_list2.size(); i++)
		{
			string feature = feature_list2[i]; 
			std::ifstream feature_in(feature, std::ios::in);
			float value;
			for (int i = 0; i<feature_dim; i++)
			{
				feature_in >> value;
				merge_datum.add_float_data(value);
			}
			feature_in.close();
		}
		merge_datum.SerializeToString(&out);
		int length = sprintf_s(key, kMaxKeyLength, "%08d", pairid);
		batch->Put(string(key, length), out);
		if (pairid % 100 == 0)
		{
			LOG(INFO) << "pairid:" << pairid;
			db->Write(leveldb::WriteOptions(), batch);
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}

	delete batch;
	delete db;
}


int main(int argc, char** argv)
{
	if (argc != 6)
	{
		printf("Usage:\n"
			"    EXE video2feature_list video_pair_list output_db_file feature_dim RANDOM_SHUFFLE_DATA[0 or 1]"
			"\n"
			);
	}
	else {
		google::InitGoogleLogging(argv[0]);
		int feature_dim= atoi(argv[4]);
		int shuffle = atoi(argv[5]);
		convert_dataset(argv[1], argv[2], argv[3], feature_dim, shuffle);
	}
	return 0;
}
