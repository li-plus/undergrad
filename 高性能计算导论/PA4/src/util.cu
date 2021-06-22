#include "../include/util.h"

int kNumV = -1, kNumE = -1, kLen = 0;
vector<void *> registered_ptr;
size_t total_size = 0;
string basedir = "";
string inputgraph = "";
string edgefile = "";
string ptrfile = "";
string dataset = "";

curandGenerator_t kCuRand;

void argParse(int argc, char ** argv, int* p_limit, int* p_limit2)
{
    args::ArgumentParser parser("GNN parameters", "");
    args::ValueFlag<string> arg_dataset(parser, "dataset", "", {"dataset"});
    args::ValueFlag<string> arg_datadir(parser, "datadir", "", {"datadir"});

    args::ValueFlag<int> arg_len(parser, "len", "", {"len"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        exit(0);
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    if(bool{arg_datadir})
        basedir = args::get(arg_datadir);
    else
        basedir = "/home/course/hpc/assignments/2021/data/public/PA4/";
    assert(bool{arg_dataset});
    string dset = args::get(arg_dataset);
    dataset = dset;
    inputgraph = basedir + dset + ".graph";
    ptrfile = inputgraph + ".ptrdump";
    edgefile = inputgraph + ".edgedump";
    string configpath = basedir + dset + ".config";
    assert(fexist(configpath));
    FILE *fin(fopen(configpath.c_str(), "r"));
    fscanf(fin, "%d", &kNumV);
    fscanf(fin, "%d", &kNumE);
    fclose(fin);
    if(fexist(inputgraph))
    {
        if(!fexist(ptrfile)) ptrfile = "";
        if(!fexist(edgefile)) edgefile = "";
    }
    else
    {
        assert(fexist(ptrfile));
        assert(fexist(edgefile));
    }
    
    assert(bool{arg_len});
    kLen = args::get(arg_len);
    
    dbg(dset);
    inputgraph = dset;
}


// ************************************************************
// variables for single train
int *gptr, *gidx;
