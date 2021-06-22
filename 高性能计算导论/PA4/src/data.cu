#include "../include/data.h"

void load_graph(std::string dset, int &num_v, int &num_e, int* &indptr, int* &indices)
{
    dbg("loading");
    auto inputgraph = basedir + dset + ".graph";
    auto ptrfile = inputgraph + ".ptrdump";
    auto edgefile = inputgraph + ".edgedump";
    auto configpath = basedir + dset + ".config";

    assert(fexist(configpath));
    FILE *fin(fopen(configpath.c_str(), "r"));
    fscanf(fin, "%d", &num_v);
    fscanf(fin, "%d", &num_e);
    fclose(fin);

    indptr = new int[num_v + 1];
    indices = new int[num_e];

    // ptr
    if(fexist(ptrfile))
    {
        FILE *f1(fopen(ptrfile.c_str(), "r"));
        fread(indptr, (num_v + 1) * sizeof(int), 1, f1);
        fclose(f1);
    }
    else
    {
        dbg("reading non-processed file");
        FILE *tmpfin(fopen(inputgraph.c_str(), "r"));
        fin = tmpfin;
        for (int i = 0; i < num_v + 1; ++i) {
            fscanf(fin, "%d", indptr + i);
        }
        ptrfile = inputgraph + ".ptrdump";
        FILE *f2(fopen(ptrfile.c_str(), "w"));
        fwrite((void*)indptr, (num_v + 1) * sizeof(int), 1, f2);
        fclose(f2);
    }
    if(indptr[num_v] != num_e)
    {
        dbg(indptr[num_v]);
        dbg(num_e);
        assert(indptr[num_v] == num_e);
    }

    // idx
    if(fexist(edgefile))
    {
        FILE *f2(fopen(edgefile.c_str(), "r"));
        fread(indices, num_e * sizeof(int), 1, f2);
        fclose(f2);
    }
    else
    {
        for (int i = 0; i < num_e; ++i) 
        {
            fscanf(fin, "%d", indices + i);
        }
        edgefile = inputgraph + ".edgedump";
        FILE *f2(fopen(edgefile.c_str(), "w"));
        fwrite((void*)indices, num_e * sizeof(int), 1, f2);
        fclose(f2);
        fclose(fin);
    }
}