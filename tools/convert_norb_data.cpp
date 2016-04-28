// This script converts the MNIST dataset to a lmdb (default) or
// leveldb (--backend=leveldb) format used by caffe to load data.
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <iostream>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"

// port for Win32
#ifdef _MSC_VER
#include <direct.h>
#define snprintf sprintf_s
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

const int kNORBImgNBytes = 4608;

DEFINE_string(backend, "leveldb", "The backend for storing the result");
DEFINE_bool(build_valid, false, "If we need to split training set into new train and valid");

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

//read label and float data into datum
void read_image_float(std::ifstream* file, int* label, Datum* datum){
	datum->clear_data();
	datum->clear_float_data();
	char label_char;
	file->read(&label_char, 1);
	datum->set_label(label_char);
	const int size_float = sizeof(float);
	//float32 is 4 bytes per data
	for (int i = 0; i < kNORBImgNBytes; i++){
		float value;
		file->read((char*)&value, size_float);
		datum->add_float_data(value);
	}
}

void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_path, const string& db_backend) {
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  // lmdb
  MDB_env *mdb_env=NULL;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn=NULL;
  // leveldb
  leveldb::DB* db=NULL;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_path
        << ". Is it already existing?";
    batch = new leveldb::WriteBatch();
  } else if (db_backend == "lmdb") {  // lmdb
    LOG(INFO) << "Opening lmdb " << db_path;
	// port for Win32
#ifndef _MSC_VER
	CHECK_EQ(mkdir(db_path, 0744), 0) 
#else
	CHECK_EQ(_mkdir(db_path), 0) 
#endif
        << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed. Does the lmdb already exist? ";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  // Storing to db
  char label;
  char* pixels = new char[rows * cols];
  int count = 0;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  string value;

  Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    datum.set_data(pixels, rows*cols);
    datum.set_label(label);
    snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
    datum.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    if (db_backend == "leveldb") {  // leveldb
      batch->Put(keystr, value);
    } else if (db_backend == "lmdb") {  // lmdb
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = keystr.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
          << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    if (++count % 1000 == 0) {
      // Commit txn
      if (db_backend == "leveldb") {  // leveldb
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        batch = new leveldb::WriteBatch();
      } else if (db_backend == "lmdb") {  // lmdb
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    if (db_backend == "leveldb") {  // leveldb
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      delete db;
    } else if (db_backend == "lmdb") {  // lmdb
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      mdb_close(mdb_env, mdb_dbi);
      mdb_env_close(mdb_env);
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }
    LOG(ERROR) << "Processed " << count << " files.";
  }
  delete pixels;
}


void convert_dataset_raw(const char* folder_name, const char* db_path, 
	const int start_batch_id, const int end_batch_id,
	string batch_prefix, const string& db_backend) {

	CHECK_GE(end_batch_id, start_batch_id);
	// lmdb
	MDB_env *mdb_env = NULL;
	MDB_dbi mdb_dbi;
	MDB_val mdb_key, mdb_data;
	MDB_txn *mdb_txn = NULL;
	// leveldb
	leveldb::DB* db = NULL;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	leveldb::WriteBatch* batch = NULL;

	// Open db
	if (db_backend == "leveldb") {  // leveldb
		LOG(INFO) << "Opening leveldb " << db_path;
		leveldb::Status status = leveldb::DB::Open(
			options, db_path, &db);
		CHECK(status.ok()) << "Failed to open leveldb " << db_path
			<< ". Is it already existing?";
		batch = new leveldb::WriteBatch();
	}
	else if (db_backend == "lmdb") {  // lmdb
		LOG(INFO) << "Opening lmdb " << db_path;
		// port for Win32
#ifndef _MSC_VER
		CHECK_EQ(mkdir(db_path, 0744), 0) 
#else
		CHECK_EQ(_mkdir(db_path), 0)
#endif
			<< "mkdir " << db_path << "failed";
		CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
		CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
			<< "mdb_env_set_mapsize failed";
		CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
			<< "mdb_env_open failed";
		CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
			<< "mdb_txn_begin failed";
		CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
			<< "mdb_open failed. Does the lmdb already exist? ";
	}
	else {
		LOG(FATAL) << "Unknown db backend " << db_backend;
	}
	
	//image meta data
	uint32_t rows = 32;
	uint32_t cols = 32;
	uint32_t num_items = 128000;
	uint32_t num_channels = 3;
	// db buffer
	int label;
	float* pixels = new float[rows * cols * num_channels];
	int count = 0;
	const int kMaxKeyLength = 15;
	char key_cstr[kMaxKeyLength];
	char index_cstr[kMaxKeyLength];
	string path_gap = "\\";
	bool is_binary = false;
	const int size_float = sizeof(float);

	for (int batch_id = start_batch_id; batch_id < end_batch_id + 1; batch_id++){
		if (batch_id == 5){
			num_items = 92388;
		}
		else if (batch_id == 6){
			num_items = 26032;
		}
		// Open files
		snprintf(index_cstr, kMaxKeyLength, "_%1d", batch_id);
		string idx_str(index_cstr);
		string zcn_str("--zcn");
		string image_filename = folder_name + path_gap + batch_prefix + idx_str;
		std::ifstream image_file(image_filename, std::ios::in);
		CHECK(image_file) << "Unable to open file " << image_filename;
		LOG(INFO) << "load data in " << image_filename;
		string value;
		Datum datum;
		datum.set_channels(num_channels);
		datum.set_height(rows);
		datum.set_width(cols);
		LOG(INFO) << "A total of " << num_items << " items.";
		LOG(INFO) << "Rows: " << rows << " Cols: " << cols << " Channels: " << num_channels;
		for (int item_id = 0; item_id < num_items; ++item_id) {
			datum.clear_float_data();
			datum.clear_data();
			//read image data
			if (is_binary){
				read_image_float(&image_file, &label, &datum);
			}
			else{
				image_file >> label;
				datum.set_label(label);
				for (int p = 0; p < cols * rows * num_channels; p++){
					image_file >> pixels[p];
					datum.add_float_data(pixels[p]);
				}
			}
			int tmp_label = datum.label();
			CHECK_EQ(datum.float_data_size(), rows * cols * num_channels)
				<< "loaded data size not match with given data size";
			snprintf(key_cstr, kMaxKeyLength, "%02d_%08d", batch_id, item_id);
			datum.SerializeToString(&value);
			string keystr(key_cstr);

			// Put in db
			if (db_backend == "leveldb") {  // leveldb
				batch->Put(keystr, value);
			}
			else if (db_backend == "lmdb") {  // lmdb
				mdb_data.mv_size = value.size();
				mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
				mdb_key.mv_size = keystr.size();
				mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
				CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
					<< "mdb_put failed";
			}
			else {
				LOG(FATAL) << "Unknown db backend " << db_backend;
			}

			if (++count % 1000 == 0) {
				// Commit txn
				if (db_backend == "leveldb") {  // leveldb
					db->Write(leveldb::WriteOptions(), batch);
					delete batch;
					batch = new leveldb::WriteBatch();
				}
				else if (db_backend == "lmdb") {  // lmdb
					CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
						<< "mdb_txn_commit failed";
					CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
						<< "mdb_txn_begin failed";
				}
				else {
					LOG(FATAL) << "Unknown db backend " << db_backend;
				}
			}
		}
		// write the last batch
		if (count % 1000 != 0) {
			if (db_backend == "leveldb") {  // leveldb
				db->Write(leveldb::WriteOptions(), batch);
//				delete batch;
//				delete db;
			}
			else if (db_backend == "lmdb") {  // lmdb
				CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
//				mdb_close(mdb_env, mdb_dbi);
//				mdb_env_close(mdb_env);
			}
			else {
				LOG(FATAL) << "Unknown db backend " << db_backend;
			}
			LOG(ERROR) << "Processed " << count << " files.";
		}
	}
	delete pixels;
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("This script converts the MNIST dataset to\n"
        "the lmdb/leveldb format used by Caffe to load data.\n"
        "Usage:\n"
        "    convert_mnist_data [FLAGS] input_image_file input_label_file "
        "output_db_file\n"
        "or    convert_mnist_data [FLAGS] input_folder_path output_db_file\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading,"
        "or directly use data/mnist/get_mnist.sh\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  //also log to console window
  FLAGS_alsologtostderr = 1;

  const string& db_backend = FLAGS_backend;

  if (argc != 4 && argc != 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/mnist/convert_mnist_data");
  } else {
    google::InitGoogleLogging(argv[0]);
	if (argc == 4){
		convert_dataset(argv[1], argv[2], argv[3], db_backend);
	}
	else{
		convert_dataset_raw(argv[1], argv[2], 6, 6, "data_batch_float", db_backend);
	}
  }
  return 0;
}
