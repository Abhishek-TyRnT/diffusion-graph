#ifndef DIFFUSION_GRAPH_GRAPH_WRITER_H
#define DIFFUSION_GRAPH_GRAPH_WRITER_H

#include <unordered_map>
#include <variant>
#include <any>
#include <fstream>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "weightBuffers.pb.h"

using NestedValueType = std::variant<std::string, 
                            std::vector<int64_t>, 
                            std::vector<std::string>, 
                            int, float>;
using ValueType = std::variant<std::string, 
                    std::unordered_map<std::string, NestedValueType>, std::vector<std::string>>;

using SubGraphMap = std::unordered_map<std::string, ValueType>;

class GraphWriter{
/*The Graph Writer class converts the vllm_graph IR to unordered map to subsequently 
    convert to json format.*/

private:
    
    struct WeightShard {
        DenseWeights::WeightsData data;
        std::string shard_path;
        std::string shard_name;
        size_t space_occupied = 0;

        WeightShard(std::string directory, std::string shard_name) : 
            shard_path(directory + "/" + shard_name), shard_name(shard_name) {}
        
        bool isSpaceAvailable(size_t requested_space){
            return (space_occupied + requested_space) < 2e+9;
        }

        void updateSpaceOccupied(size_t requested_space){
            space_occupied += requested_space;
        }

        std::string getShardName(){
            return shard_name;
        }

        std::string getShardPath(){
            return shard_path;
        }

        DenseWeights::WeightsData& getWeightsData(){
            return data;
        }
        void saveShard(){
            std::ofstream output(shard_path, std::ios::binary);
            data.SerializeToOstream(&output);
            output.close();
        }
    };
    std::unordered_map<std::string, std::variant<SubGraphMap, std::string>> graph;
    llvm::DenseMap<mlir::Value, std::string> opMap;
    
    //This one is kept to store other Dense Elements and Scalars
    DenseWeights::WeightsData constData;
    std::vector<WeightShard> DenseResourceData;
    //Counter to keep number of shards
    int shard_index = 0;
    std::string weightsPath;
    //Counter to keep number of ops
    uint64_t opCount = 0;
    //Counter to keep number of args
    uint64_t argCount = 0; 
    //Counter to keep number of funcs
    uint64_t funcCount = 0;
    void addOp(mlir::Operation *op, SubGraphMap &subGraph);
    template<typename ElemType>
    void storeWeights(ElemType val, std::string ssa_id);

public:
    
    void build(mlir::OwningOpRef<mlir::ModuleOp> &module, 
               llvm::DenseMap<mlir::StringRef, mlir::ArrayRef<char>> &dialectResourcesMap);
    GraphWriter(std::string weightsPath);
    void closeFile();
    std::unordered_map<std::string, std::variant<SubGraphMap, std::string>> getGraph(){ return graph; }
};
#endif


