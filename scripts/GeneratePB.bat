if exist "../src/caffe/proto/caffe.pb.h" (
    echo caffe.pb.h remains the same as before
) else (
    echo caffe.pb.h is being generated
    "../3rdparty/bin/protoc" -I="../src/caffe/proto" --cpp_out="../src/caffe/proto" "../src/caffe/proto/caffe.proto"
)

REM if exist "../src/caffe/proto/caffe_pretty_print.pb.h" (
    REM echo caffe_pretty_print.pb.h remains the same as before
REM ) else (
    REM echo caffe_pretty_print.pb.h is being generated
    REM "../3rdparty/bin/protoc" -I="../src/caffe/proto" --cpp_out="../src/caffe/proto" "../src/caffe/proto/caffe_pretty_print.proto"
REM )


