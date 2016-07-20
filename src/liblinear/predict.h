#include "linear.h"
#ifndef PREDICT_H_
#define PREDICT_H_

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <vector>

float liblinear_predict(FILE* input, FILE* output, const char* model_file,
	bool pred_prob);

#endif 
