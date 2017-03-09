#include "caffe/util/file_proc_util.h"


int LoadPathFromFile(string imgPathFile, vector<string> &imgPath)
{
	imgPath.clear();
	std::ifstream  inImgPath(imgPathFile.c_str());
	if (!inImgPath.is_open()){
		printf("can not open file %s\n", imgPathFile.c_str());
		return -1;
	}
	string imgName;
	while (inImgPath.good())
	{
		getline(inImgPath, imgName);
		if (imgName.length() == 0)
		{
			break;
		}
		imgPath.push_back(imgName);
	}
	if (imgPath.size() == 0)
	{
		printf("Load image file %s failed!\n", imgPathFile.c_str());
		return 0;
	}
	return 1;
}

int ListFilesInDir(string folder_path, vector<string>& fileList){
	fileList.clear();
	if (!folder_path.empty()){
		namespace fs = boost::filesystem;
		fs::path apk_path(folder_path);
		fs::recursive_directory_iterator end;
		for (fs::recursive_directory_iterator i(apk_path); i != end; ++i){
			const fs::path cp = (*i);
			if (fs::is_regular_file(cp)){
				fileList.push_back(cp.string());
			}
		}
		return 0;
	}
	return 1;
}

/*
* 0: not exist
* 1: not a regular file or not a directory
* 2: file
* 3: directory
*/
int check_file_type(string file_name){
	namespace fs = boost::filesystem;
	fs::path p(file_name);
	if (fs::exists(p)){
		if (fs::is_regular_file(p)){
			return 2;
		}
		else if (fs::is_directory(p)){
			return 3;
		}
		else{
			return 1;
		}
	}
	else{
		return 0;
	}
}

string get_ext(string file_name){
	size_t p = file_name.rfind('.');
	std::string ext = p != file_name.npos ? file_name.substr(p) : file_name;
	return ext;
}