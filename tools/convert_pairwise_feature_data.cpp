//added by qing li 2014-12-27
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"


void read_feature(std::pair<int, int> feature_pair, std::vector<std::string> feature_list, int dim, caffe::Datum& datum) {
	
  std::ifstream feature1(feature_list[feature_pair.first], std::ios::in);
  std::ifstream feature2(feature_list[feature_pair.second], std::ios::in);
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
  feature1.close();
  feature2.close();
}

void convert_dataset(const char* feature_list_filename, const char* pair_list_filename,
        const char* db_filename, int feature_dim, int shuffle) {
  // Open files
  std::ifstream feature_list_in(feature_list_filename, std::ios::in);
  std::ifstream pair_list_in(pair_list_filename, std::ios::in);
  CHECK(feature_list_in) << "Unable to open file " << feature_list_filename;
  CHECK(pair_list_in) << "Unable to open file " << pair_list_filename;
  
  //read metadata
  std::vector<std::string> feature_list;
  std::string feature;
  while(feature_list_in>>feature)
  {
    if(feature.size()!=0)
		feature_list.push_back(feature);
  }

  int index1;
  int index2;

  std::vector<std::pair<int, int>> pair_list;
  while(pair_list_in>>index1>>index2)
  {
    pair_list.push_back(std::make_pair(index1, index2));
    //LOG(INFO)<<"INDEX:"<<index1<<'\t'<<index2;
  }

  feature_list_in.close();
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


  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;

  
  
  for (int pairid = 0; pairid < num_pairs; ++pairid) {
    caffe::Datum datum;
    datum.set_channels(2*feature_dim);  // one channel for each image in the pair
    datum.set_height(1);
    datum.set_width(1);
    read_feature(pair_list[pairid], feature_list, feature_dim, datum);
    
    datum.set_label(1);
    datum.SerializeToString(&value);
    _snprintf(key, kMaxKeyLength, "%08d", pairid);
    db->Put(leveldb::WriteOptions(), std::string(key), value);
	if(pairid%100==0)
		LOG(INFO)<<"pairid:"<<pairid;
  }

  delete db;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf("This script converts the MNIST dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
           "    convert_mnist_data feature_list pair_list output_db_file feature_dim RANDOM_SHUFFLE_DATA[0 or 1]"
           "\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    int feature_dim=atoi(argv[4]);
	int shuffle=atoi(argv[5]);
    convert_dataset(argv[1], argv[2], argv[3], feature_dim, shuffle);
  }
  return 0;
}
