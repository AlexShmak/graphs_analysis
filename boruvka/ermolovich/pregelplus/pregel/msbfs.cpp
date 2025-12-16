#include "basic/pregel-dev.h"
#include <unordered_map>
using namespace std;

#define NO_PARENT -1
#define ROOT      -2

struct PBFSValue {
    vector<VertexID> parents;
    vector<VertexID> edges;
};

obinstream& operator>>(obinstream& m, PBFSValue& v) {
    m >> v.parents;
    m >> v.edges;
    return m;
}

ibinstream& operator<<(ibinstream& m, const PBFSValue& v) {
    m << v.parents;
    m << v.edges;
    return m;
}

struct PBFSMessage {
    int startGroup;
    VertexID parent;
};

ibinstream& operator<<(ibinstream& m, const PBFSMessage& v) {
    m << v.startGroup;
    m << v.parent;
    return m;
}

obinstream& operator>>(obinstream& m, PBFSMessage& v) {
    m >> v.startGroup;
    m >> v.parent;
    return m;
}


class PBFSVertex : public Vertex<VertexID, PBFSValue, PBFSMessage> {
  public:
    void broadcast(PBFSMessage msg, VertexID exception) {
        vector<VertexID>& nbs = value().edges;
        for (int i = 0; i < nbs.size(); i++) {
            if (nbs[i] != exception)
                send_message(nbs[i], msg);
        }
    }

    virtual void compute(MessageContainer& messages) {
        if (step_num() == 1) {
            auto parents = value().parents;
            for (int sg = 0; sg < parents.size(); sg++) {
                VertexID par = parents[sg];
                if (par == ROOT)
                    broadcast({sg, id}, id);
            }
        } else {
            std::sort(messages.begin(), messages.end(), [](const PBFSMessage& a, const PBFSMessage& b) {
                if (a.startGroup != b.startGroup) {
                    return a.startGroup < b.startGroup;
                }
                return a.parent < b.parent;
            });
            int prevStartGroup = -1;
            for (size_t i = 0; i < messages.size(); i++) {
                if (messages[i].startGroup != prevStartGroup) {
                    int sg = messages[i].startGroup;
                    if (value().parents[sg] == NO_PARENT) {
                        int par = messages[i].parent;

                        value().parents[sg] = par;
                        broadcast({sg, id}, par);
                        prevStartGroup = sg;
                    }
                }
            }
            vote_to_halt();
        }
    }
};

class PBFSWorker : public Worker<PBFSVertex> {
    char buf[256];
    size_t numOfStartVertexes = -1;
    vector<VertexID> startVertexes;

  public:
    virtual PBFSVertex* toVertex(char* line) {

        char* pch;
        pch = strtok(line, "\t");
        PBFSVertex* v = new PBFSVertex;
        if (strcmp(pch, "p") == 0) {
            v->id = -1;
            pch = strtok(NULL, " ");
            numOfStartVertexes = atoi(pch);
            for (int i = 0; i < numOfStartVertexes; i++) {
                pch = strtok(NULL, " ");
                startVertexes.push_back(atoi(pch));
            }
        } else {
            v->id = atoi(pch);
            v->value().parents = vector(numOfStartVertexes, NO_PARENT);
            pch = strtok(NULL, " ");
            int num = atoi(pch);
            for (int i = 0; i < num; i++) {
                pch = strtok(NULL, " ");
                v->value().edges.push_back(atoi(pch));
            }

            auto iter = find(startVertexes.begin(), startVertexes.end(), v->id);
            if (iter != startVertexes.end()) {
                int start_group = iter - startVertexes.begin();
                v->value().parents[start_group] = ROOT;
            }
        }
        return v;
    }

    virtual void toline(PBFSVertex* v, BufferedWriter& writer) {
        if (v->id < 0)
            return;

        char* buf_ptr = buf;
        buf_ptr += sprintf(buf_ptr, "%d\t", v->id);
        PBFSValue value = v->value();
        for (size_t i = 0; i < value.parents.size(); i++)
            buf_ptr += sprintf(buf_ptr, "%d ", value.parents[i]);
        buf_ptr += sprintf(buf_ptr, "\n");

        writer.write(buf);
    }
};

void req_pbfs(string in_path, string out_path) {
    WorkerParams param;
    param.input_path = in_path;
    param.output_path = out_path;
    param.force_write = true;
    param.native_dispatcher = false;
    PBFSWorker worker;
    worker.run(param);
}

int main(int argc, char* argv[]) {
    init_workers();
    req_pbfs("/toyFolder", "/toyOutput");
    worker_finalize();
    return 0;
}