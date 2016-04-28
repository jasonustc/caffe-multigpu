//added by qing li 2014-12-27
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

using namespace std;
using namespace caffe;


void read_feature(std::pair<int, int> feature_pair, std::vector<std::string> video2img_list, int dim, caffe::Datum& datum) {
	
  std::ifstream feature1(video2img_list[feature_pair.first], std::ios::in);
  std::ifstream feature2(video2img_list[feature_pair.second], std::ios::in);
  float value;
  for(int i=0;i<dim;i++)
  {
    feature1>>value;
	datum.add_float_data(value);
  }

  for(int i=dim;i<2*dim;i++)
  {
    feature2>>value;
	datum.add_float_data(value);
  }
}

void convert_dataset(const char* video2img_list_filename, const char* pair_list_filename,
        const char* db_filename, int resize_width, int resize_height, int shuffle) {
  // Open files
  std::ifstream video2img_list_in(video2img_list_filename, std::ios::in);
  std::ifstream pair_list_in(pair_list_filename, std::ios::in);
  CHECK(video2img_list_in) << "Unable to open file " << video2img_list_filename;
  CHECK(pair_list_in) << "Unable to open file " << pair_list_filename;
  
  //read metadata
  map<int, vector<string>> video2img_list;
  int video;
  string img;
  while(video2img_list_in>>video>>img)
  {
	  if (video2img_list.find(video) == video2img_list.end())
		  video2img_list.insert(pair<int, vector<string>>(video, vector<string>()));
	  video2img_list[video].push_back(img);
  }

  int index1;
  int index2;

  std::vector<std::pair<int, int>> pair_list;
  while(pair_list_in>>index1>>index2)
  {
    pair_list.push_back(std::make_pair(index1, index2));
    //LOG(INFO)<<"INDEX:"<<index1<<'\t'<<index2;
  }

  video2img_list_in.close();
  pair_list_in.close();
 

  if(shuffle==1)
  {
	  LOG(INFO)<<"Shuffling data";
	  std::random_shuffle(pair_list.begin(), pair_list.end());
  }
   int num_pairs=pair_list.size();

  LOG(INFO)<<"A total of "<<num_pairs<<" pairs";

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";


  const int kMaxKeyLength = 256;
  char key[kMaxKeyLength];
  std::string value;

  for (int pairid = 0; pairid < num_pairs; ++pairid) {
	  Datum merge_datum;
	  vector<string> img_list1 = video2img_list[pair_list[pairid].first];
	  vector<string> img_list2 = video2img_list[pair_list[pairid].second];
	  CHECK(img_list1.size() == img_list2.size()) << "the pair have different num of images" << '\n';
	  merge_datum.set_channels(img_list1.size() * 3 * 2);
	  merge_datum.set_width(resize_width);
	  merge_datum.set_height(resize_height);
	  merge_datum.set_encoded(true);
	  //char* buffer = new char[merge_datum.channels()*merge_datum.width()*merge_datum.height()];
	  string buffer;
	  for (int i = 0; i < img_list1.size(); i++)
	  {
		  string img = img_list1[i];
		  Datum one_img_datum;
		  bool status;
		  std::string enc = "jpg";
		  status = ReadImageToDatum(img, 1, resize_height, resize_width, true,
			  enc, &one_img_datum);
		  CHECK(status == true) << "fail to read image:" << img << '\n';
		  const string& data = one_img_datum.data();
		  buffer.append(data);
	  }
	  for (int i = 0; i < img_list2.size(); i++)
	  {
		  string img = img_list2[i];
		  Datum one_img_datum;
		  bool status;
		  std::string enc = "jpg";
		  status = ReadImageToDatum(img, 1, resize_height, resize_width, true,
			  enc, &one_img_datum);
		  CHECK(status == true) << "fail to read image:" << img << '\n';
		  const string& data = one_img_datum.data();
		  buffer.append(data);
	  }
	  CHECK(buffer.size() == merge_datum.channels()*merge_datum.width()*merge_datum.height()) << "buffer != merge_datum\n";
	  merge_datum.set_data(buffer);
	  merge_datum.SerializeToString(&value);
	  int length = sprintf_s(key, kMaxKeyLength, "%08d", pairid);
	  db->Put(leveldb::WriteOptions(), string(key, length), value);
	  if (pairid % 100==0)
		  LOG(INFO) << "pairid:" << pairid;
  }
    
  delete db;
}


int main(int argc, char** argv)
{
	if (argc!=7)
	{
    printf("Usage:\n"
           "    EXE video2img_list video_pair_list output_db_file img_width img_height RANDOM_SHUFFLE_DATA[0 or 1]"
           "\n"
           );
	}
	else {
		google::InitGoogleLogging(argv[0]);
		int width = atoi(argv[4]);
		int height = atoi(argv[5]);
		int shuffle = atoi(argv[6]);
		convert_dataset(argv[1], argv[2], argv[3], width, height, shuffle);
	}
    return 0;
}
