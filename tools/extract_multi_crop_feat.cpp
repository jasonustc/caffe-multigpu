#include <iostream>
#include <fstream>
#include <string>

#include "caffe/fea_extractor.h"

using namespace std;

void load_file_list(const string& file_path, vector<string>& imgList){
	imgList.clear();
	ifstream inImages(file_path);
	if (!inImages.is_open()){
		LOG(FATAL) << "can not load indexes from file " << file_path;
	}
	string imgPath;
	while (inImages >> imgPath){
		imgList.push_back(imgPath);
	}
	LOG(INFO) << "Load " << imgList.size() << " images.";
}

bool is_txt(const string& file_path){
	string ext = file_path.substr(file_path.rfind('.') + 1);
	return ext == "txt";
}

bool load_coeffs(const string& file_path,float* coeffs, const int dim = 4096){
	ifstream inCoeff(file_path.c_str());
	CHECK(inCoeff.is_open()) << "Failed to open coefficients file " << file_path.c_str();
	int i = 0;
	float feat;
	while (inCoeff >> feat){
		coeffs[i] = feat;
		i++;
	}
	CHECK_EQ(i, dim) << "Feature dim not match with #coeffs in file " << file_path.c_str();
	inCoeff.close();
	return true;
}

void NormalizeFeat(const int count, float* feat, const float* coeffs){
	//divided by max value in each dim
	caffe::caffe_div<float>(count, feat, coeffs, feat);
	//sqare
	caffe::caffe_sqr(count, feat, feat);
	//sum of square
	float sqr_sum = caffe::caffe_cpu_asum(count, feat);
	//normalize
	caffe::caffe_scal<float>(count, (float)1. / sqr_sum, feat);
}

int main(int argc, char** argv) {
	if (argc < 7){
		LOG(FATAL) << "Usage: extract_features model_path net_path "
			<< "mean_path layer_name img_index_path(txt file or jpg file)";
		return 0;
	}

	string model_path = argv[1];
	string net_path = argv[2];
	string mean_path = argv[3];
	string layer_name = argv[4];
	string img_path = argv[5];
	string coeff_path = argv[6];
	vector<string> img_list;
	bool txt = is_txt(img_path);
	if (txt){
		load_file_list(img_path, img_list);
		CHECK_GT(img_list.size(), 0) << "no images in the txt file";
	}
	

	FeaExtractor<float>* fea_extractor = NULL;
	bool succ = CreateHandle(fea_extractor);
	if (!succ)
	{
		cout << "error, can not create handle" << endl;
		return -1;
	}

	succ = LoadDModel(fea_extractor, mean_path, net_path, model_path, layer_name); // load vgg model
	if (!succ)
	{
		cout << "extractor failed to load model" << endl;
		return -1;
	}

	int dim = GetFeaDim(fea_extractor); // get the dimension of features
	float* coeffs = new float[dim];
	int channel = 3;

	if (txt){
		for (size_t i = 0; i < img_list.size(); i++)
		{
			ExtractFeaturesByPath(
				fea_extractor,
				img_list[i]);

			float *fea = GetMutableFeaPtr(fea_extractor);
//			load_coeffs(coeff_path, coeffs, dim);
//			NormalizeFeat(dim, fea, coeffs);
			string out_feat_path = img_list[i] + ".feat";
			ofstream outfile(out_feat_path);

			for (int j = 0; j < dim; j++)
			{
				outfile << fea[j] << " ";
			}
			outfile << endl;
			outfile.close();
		}
	}
	else{
		string output_path = img_path + ".feat";
		ofstream outfile(output_path);
		ExtractFeaturesByPath(fea_extractor, img_path);
		float *fea = GetMutableFeaPtr(fea_extractor);
		load_coeffs(coeff_path, coeffs, dim);
		NormalizeFeat(dim, fea, coeffs);
		for (int j = 0; j < dim; j++)
		{
			outfile << fea[j] << " ";
		}
		outfile << endl;
		outfile.close();
	}

	delete[] coeffs;
	ReleaseHandle(fea_extractor);
	return 0;
}