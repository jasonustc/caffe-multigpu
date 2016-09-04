#ifndef FILE_PROC_UTIL_H_
#define FILE_PROC_UTIL_H_

#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>

using std::string;
using std::vector;

/*
 * @param imagPathFile: one line per image
 */
int LoadPathFromFile(string imgPathFile, vector<string> &imgPath);

int ListFilesInDir(string folder_path, vector<string>& fileList);

/*
 * 0: not exist
 * 1: not a regular file or not a directory
 * 2: file
 * 3: directory
 */
int check_file_type(string file_name);

/*
 * e.g. "image.jpg" will return ".jpg"
 */
string get_ext(string file_name);

#endif