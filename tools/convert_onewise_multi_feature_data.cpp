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


void convert_dataset(const char* video2feature_list_filename, const char* video_list_filename,
	const char* db_filename, int feature_dim, int shuffle) {
	// Open files
	std::ifstream video2feature_list_in(video2feature_list_filename, std::ios::in);
	std::ifstream video_list_in(video_list_filename, std::ios::in);
	CHECK(video2feature_list_in) << "Unable to open file " << video2feature_list_filename;
	CHECK(video_list_in) << "Unable to open file " << video_list_filename;

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
	std::vector<int> video_list;
	while (video_list_in >> index1 )
	{
		video_list.push_back(index1);
		//LOG(INFO)<<"INDEX:"<<index1<<'\t'<<index2;
	}

	video2feature_list_in.close();
	video_list_in.close();


	if (shuffle == 1)
	{
		LOG(INFO) << "Shuffling data";
		std::random_shuffle(video_list.begin(), video_list.end());
	}
	int num_videos= video_list.size();

	LOG(INFO) << "A total of " << num_videos << " videos";

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

	for (int videoid = 0; videoid < num_videos; ++videoid) {
		Datum merge_datum;
		vector<string> feature_list1 = video2feature_list[video_list[videoid]];
		merge_datum.set_channels(feature_list1.size() * feature_dim );
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
		merge_datum.SerializeToString(&out);
		int length = sprintf_s(key, kMaxKeyLength, "%08d", videoid);
		batch->Put(string(key, length), out);
		if (videoid% 100 == 0)
		{
			LOG(INFO) << "videoid:" << videoid;
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
			"    EXE video2feature_list video_video_list output_db_file feature_dim RANDOM_SHUFFLE_DATA[0 or 1]"
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
