//===-- ExecutionState.cpp ------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExecutionState.h"

#include "Memory.h"

#include "klee/Expr/Expr.h"
#include "klee/Module/Cell.h"
#include "klee/Module/InstructionInfoTable.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"
#include "klee/Support/Casting.h"
#include "klee/Support/OptionCategories.h"

#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"

#include <cassert>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <regex>
#include <sstream>
#include <stdarg.h>
#include <iostream>

using namespace llvm;
using namespace klee;

namespace {
  cl::opt<bool> DebugLogStateMerge(
      "debug-log-state-merge", cl::init(false),
      cl::desc("Debug information for underlying state merging (default=false)"),
      cl::cat(MergeCat));
  
}
namespace klee {
  extern cl::opt<bool> SingleObjectResolution; 
}

/***/

std::uint32_t ExecutionState::nextID = 1;

/***/

StackFrame::StackFrame(KInstIterator _caller, KFunction *_kf)
  : caller(_caller), kf(_kf), callPathNode(0), 
    minDistToUncoveredOnReturn(0), varargs(0) {
  locals = new Cell[kf->numRegisters];
}

StackFrame::StackFrame(const StackFrame &s) 
  : caller(s.caller),
    kf(s.kf),
    callPathNode(s.callPathNode),
    allocas(s.allocas),
    minDistToUncoveredOnReturn(s.minDistToUncoveredOnReturn),
    varargs(s.varargs) {
  locals = new Cell[s.kf->numRegisters];
  for (unsigned i=0; i<s.kf->numRegisters; i++)
    locals[i] = s.locals[i];
}

StackFrame::~StackFrame() { 
  delete[] locals; 
}

/***/

ExecutionState::ExecutionState(KFunction *kf, MemoryManager *mm)
    : pc(kf->instructions), prevPC(pc) {
  pushFrame(nullptr, kf);
  setID();
  if (mm->stackFactory && mm->heapFactory) {
    stackAllocator = mm->stackFactory.makeAllocator();
    heapAllocator = mm->heapFactory.makeAllocator();
  }
}

ExecutionState::~ExecutionState() {
  for (const auto &cur_mergehandler: openMergeStack){
    cur_mergehandler->removeOpenState(this);
  }

  while (!stack.empty()) popFrame();
}

ExecutionState::ExecutionState(const ExecutionState& state):
    pc(state.pc),
    prevPC(state.prevPC),
    stack(state.stack),
    incomingBBIndex(state.incomingBBIndex),
    depth(state.depth),
    addressSpace(state.addressSpace),
    stackAllocator(state.stackAllocator),
    heapAllocator(state.heapAllocator),
    constraints(state.constraints),
    pathOS(state.pathOS),
    symPathOS(state.symPathOS),
    coveredLines(state.coveredLines),
    symbolics(state.symbolics),
    cexPreferences(state.cexPreferences),
    arrayNames(state.arrayNames),
    openMergeStack(state.openMergeStack),
    steppedInstructions(state.steppedInstructions),
    instsSinceCovNew(state.instsSinceCovNew),
    unwindingInformation(state.unwindingInformation
                             ? state.unwindingInformation->clone()
                             : nullptr),
    coveredNew(state.coveredNew),
    forkDisabled(state.forkDisabled),
    base_addrs(state.base_addrs),
    base_mos(state.base_mos),
    packetRead(state.packetRead),
    packetWrite(state.packetWrite),
    mapRead(state.mapRead),
    mapWrite(state.mapWrite),
    allReads(state.allReads),
    allWrites(state.allWrites),
    argContents(state.argContents),
    mapLookupString(state.mapLookupString),
    mapLookupReturns(state.mapLookupReturns),
    callInformation(state.callInformation),
    mapMemoryObjects(state.mapMemoryObjects),
    mapCallStrings(state.mapCallStrings),
    mapCallArgumentExpressions(state.mapCallArgumentExpressions),
    branchesOnMapReturnReference(state.branchesOnMapReturnReference),
    correlatedMaps(state.correlatedMaps),
    xdpMoId(state.xdpMoId),
    nextMapName(state.nextMapName),
    nextMapKey(state.nextMapKey),
    nextMapSize(state.nextMapSize),
    nextKeySize(state.nextKeySize),
    nextValueSize(state.nextValueSize),
    mapOperationKey(state.mapOperationKey),
    generateMode(state.generateMode),
    overlap(state.overlap) {
  for (const auto &cur_mergehandler: openMergeStack)
    cur_mergehandler->addOpenState(this);
}

ExecutionState *ExecutionState::branch() {
  depth++;

  auto *falseState = new ExecutionState(*this);
  falseState->setID();
  falseState->coveredNew = false;
  falseState->coveredLines.clear();

  return falseState;
}

void ExecutionState::addPacketRead(std::string newRead) {
  if (generateMode) {
    packetRead.insert(newRead);
  } else {
    if (packetWrite.find(newRead) != packetWrite.end()) {
      overlap.insert(newRead);
    }
  }
}

void ExecutionState::addMapRead(std::string mapName, ref<Expr> key, std::string keyName) {
  auto it = mapRead.find(mapName);
  assert(key);
  if (it != mapRead.end()) {
    it->second.insert(std::make_pair(key, keyName));
  } else {
    std::set<std::pair<ref<Expr>, std::string>> newSet;
    newSet.insert(std::make_pair(key, keyName));
    mapRead.insert(std::make_pair(mapName, newSet));
  }
}

void ExecutionState::addToOverlap(std::string mapName, std::string keyValue) {
  overlap.insert("map:" + mapName + "." + keyValue);
}

void ExecutionState::addCheckRead(std::string mapName, ref<Expr> key, std::string keyName) {
  auto it = allReads.find(mapName);
  assert(key);
  if (it != allReads.end()) {
    it->second.insert(std::make_pair(key, keyName));
  } else {
    std::set<std::pair<ref<Expr>, std::string>> newSet;
    newSet.insert(std::make_pair(key, keyName));
    allReads.insert(std::make_pair(mapName, newSet));
  }
}

void ExecutionState::addCheckWrite(std::string mapName, ref<Expr> key, std::string keyName) {
  auto it = allWrites.find(mapName);
  if (it != allWrites.end()) {
    it->second.insert(std::make_pair(key, keyName));
  } else {
    std::set<std::pair<ref<Expr>, std::string>> newSet;
    newSet.insert(std::make_pair(key, keyName));
    allWrites.insert(std::make_pair(mapName, newSet));
  }
}

std::set<std::pair<ref<Expr>, std::string>> ExecutionState::getMapRead(std::string mapName) {
  std::set<std::pair<ref<Expr>, std::string>> result;
  auto it = mapRead.find(mapName);
  if (it != mapRead.end()) {
    result = it->second;
  }
  return result;
}

ref<Expr> ExecutionState::getMapReadForString(std::string mapName, std::string keyName) {
  std::set<std::pair<ref<Expr>, std::string>> result = getMapRead(mapName);
  for (auto &it : result) {
    if (keyName == it.second) {
      return it.first;
    }
  }
  auto checkIt = allReads.find(mapName);
  if (checkIt != allReads.end()) {
    result = checkIt->second;
    for (auto &it : result) {
      if (keyName == it.second) {
        return it.first;
      }
    }
  }
  llvm::errs() << "key name was " << keyName << "\n";
  assert(0 && "Failed to find key");
}

std::set<std::pair<ref<Expr>, std::string>> ExecutionState::getMapWrite(std::string mapName) {
  std::set<std::pair<ref<Expr>, std::string>> result;
  auto it = mapWrite.find(mapName);
  if (it != mapWrite.end()) {
    result = it->second;
  }
  return result;
}

void ExecutionState::addPacketWrite(std::string newWrite) {
  if (generateMode) {
    packetWrite.insert(newWrite);
  } else {
    if (packetWrite.find(newWrite) != packetWrite.end() || packetRead.find(newWrite) != packetRead.end()) {
      overlap.insert(newWrite);
    }
  }
}

void ExecutionState::addMapWrite(std::string mapName, ref<Expr> key, std::string keyName) {
  auto it = mapWrite.find(mapName);
  std::cout << "mapWrite : " << mapName << " " << keyName << "\n";
  if (it != mapWrite.end()) {
    it->second.insert(std::make_pair(key, keyName));
  } else {
    std::set<std::pair<ref<Expr>, std::string>> newSet;
    newSet.insert(std::make_pair(key, keyName));
    mapWrite.insert(std::make_pair(mapName, newSet));
  }
}

void klee::ExecutionState::addHelperFunctionCall(std::string funcName) {
  this->helperFunctions.insert(funcName);
}

std::set<std::string> ExecutionState::getReadSet() {
  std::set<std::string> readSet;
  std::string mr;
  for (auto &it : mapRead) {
    for (auto &setIt : it.second) {
      mr = "map:" + it.first + "." + setIt.second;
      readSet.insert(mr);
    }
  }

  readSet.merge(packetRead);
  return readSet;
}

std::set<std::string> ExecutionState::getReadSetMap() {
  std::set<std::string> readSetMap;
  for (auto &it : mapRead) {
    readSetMap.insert(it.first);
  }
  return readSetMap;
}

std::set<std::string> ExecutionState::getWriteSet() {
  std::set<std::string> writeSet;
  std::string mw;
  for (auto &it : mapWrite) {
    for (auto &setIt : it.second) {
      mw = "map:" + it.first + "." + setIt.second;
      writeSet.insert(mw);
    }
  }

  writeSet.merge(packetWrite);
  return writeSet;
}

std::set<std::string> ExecutionState::getWriteSetMap() {
  std::set<std::string> writeSetMap;
  for (auto &it : mapWrite) {
    writeSetMap.insert(it.first);
  }
  return writeSetMap;
}

std::set<std::string> klee::ExecutionState::getCalledHelperFunctionSet() {
  return helperFunctions;
}

bool ExecutionState::isFunctionForAnalysis(llvm::Function *func) {
  std::vector<std::string> removedFunctions = {"__uClibc_main", "__uClibc_init", "__uClibc_fini", "__user_main",
    "exit", "map_allocate", "map_lookup_elem", "map_update_elem", "map_delete_elem", 
    "map_of_map_allocate", "map_of_map_lookup_elem", "bpf_map_init_stub", "bpf_xdp_adjust_head",
    "bpf_map_lookup_elem", "bpf_map_reset_stub", "array_allocate", "bpf_map_update_elem",
    "array_update_elem", "bpf_redirect_map", "map_update_elem", "array_lookup_elem", ""};
  std::string funcName = func->getName().str();

  // not present in the removed functions
  return std::find(removedFunctions.begin(), removedFunctions.end(), funcName) == removedFunctions.end();
}

void ExecutionState::setXDPMemoryObjectID(unsigned int id) {
  xdpMoId = id;
}

unsigned int ExecutionState::getXDPMemoryObjectID() {
  return xdpMoId;
}

bool ExecutionState::isReferencetoMapReturn(llvm::Value *val) {
  for (const auto &c : callInformation) {
    if (c.second.references.find(val) != c.second.references.end()) {
      return true;
    }
  }
  return false;
}

std::vector<llvm::Value*> ExecutionState::findOriginalMapCall(llvm::Value *val) {
  std::vector<llvm::Value*> mapReturns;
  for (const auto &c : callInformation) {
    if (c.second.references.find(val) != c.second.references.end()) {
      mapReturns.push_back(c.first);
    }
  }
  return mapReturns;
}

void ExecutionState::createNewMapReturn(llvm::Value *val, const InstructionInfo *kiInfo, 
    std::string functionName, std::string mapName, std::string keyVal, std::string value) {
  std::unordered_set<const llvm::Value*> newSet;
  CallInfo info;
  newSet.insert(val);
  info.references = newSet;
  info.sourceLine = kiInfo->line;
  info.sourceColumn = kiInfo->column;
  info.sourceFile = kiInfo->file;
  info.functionName = functionName;
  info.key = keyVal;
  info.value = value;
  info.mapName = mapName;
  callInformation.insert(std::make_pair(val, info));
}

void ExecutionState::addMapString(llvm::Value *val, std::string fName, std::string mapName, std::string key, 
                                  const InstructionInfo *info, ref<Expr> keyExpr) {
  std::string mapStr = fName + " on map " + mapName + " on line: " + std::to_string(info->line) + ", column: " + std::to_string(info->column);
  mapCallStrings.insert(std::make_pair(val, std::make_pair(mapStr, key)));
  mapCallArgumentExpressions.insert(std::make_pair(val, keyExpr));
}

ref<Expr> ExecutionState::getMapCallExpr(llvm::Value *val) {
  auto key = mapCallArgumentExpressions.find(val);
  assert(key != mapCallArgumentExpressions.end());
  return key->second;
}

std::string ExecutionState::getMapCallKey(llvm::Value *val) {
  auto key = mapCallStrings.find(val);
  if (key != mapCallStrings.end()) {
    return key->second.second;
  }
  return "";
}

void ExecutionState::printReferencesToMapReturnKeys() {
  llvm::errs() << "References to map return keys: {";
  for (auto &c : callInformation) {
    c.first->dump();
  }
  llvm::errs() << "}\n";
}

bool ExecutionState::addIfReferencetoMapReturn(llvm::Value *op, llvm::Value *val) {
  bool added = false;
  for (auto &c : callInformation) {
    if (c.second.references.find(op) != c.second.references.end() || op == c.first) {
      c.second.references.insert(val);
      added = true;
    }
  }
  return added;
}

void ExecutionState::removeMapReference(llvm::Value *val) {
  for (auto &c : callInformation) {
    c.second.references.erase(val);
  }
}

// add a map correlation between source map and head map
void ExecutionState::addMapCorrelation(llvm::Value *sourceCall, llvm::Value *destCall, std::string arg) {
  correlatedMaps.insert(std::make_pair(std::make_pair(sourceCall, destCall), arg));
}

std::set<std::string> ExecutionState::formatMapCorrelations() {
  std::set<std::string> mapInfo;

  for (auto &c : correlatedMaps) {
    llvm::Value *sourceCall = c.first.first;
    llvm::Value *destCall = c.first.second;
    CallInfo sourceInfo = callInformation.find(sourceCall)->second;
    CallInfo destInfo = callInformation.find(destCall)->second;
    std::stringstream newInfo;
    newInfo << sourceInfo.sourceFile 
            << "(" << std::to_string(sourceInfo.sourceLine)
            << "," << std::to_string(sourceInfo.sourceColumn) << "):" 
            << sourceInfo.functionName << "(" 
            << sourceInfo.mapName << "," 
            << sourceInfo.key << ")->" 
            << destInfo.sourceFile 
            << "(" << std::to_string(destInfo.sourceLine) 
            << "," << std::to_string(destInfo.sourceColumn) << "):"
            << destInfo.functionName << "(" << destInfo.mapName << ",";
    if (c.second == "key") {
      newInfo << "correlation:" << destInfo.key << "," << destInfo.value << ")";
    } else if (c.second == "value") {
      newInfo << destInfo.key << "," << "correlation:" << destInfo.value << ")";
    } else {
        newInfo << destInfo.key << ")";
    }
    mapInfo.insert(newInfo.str());
  }

  return mapInfo;
}

void ExecutionState::addNewMapLookup(llvm::Value *val, std::string repr) {
  mapLookupString.insert(std::make_pair(val, repr));  
  std::unordered_set<const llvm::Value*> newSet;
  newSet.insert(val);
  mapLookupReturns.insert(std::make_pair(val, newSet));
}

bool ExecutionState::addIfMapLookupRef(llvm::Value *op, llvm::Value *val) {
  bool added = false;
  for (auto &c : mapLookupReturns) {
    if (c.second.find(op) != c.second.end() || op == c.first) {
      c.second.insert(val);
      added = true;
    }
  }
  return added;
}

std::pair<bool, std::string> ExecutionState::isMapLookupReturn(llvm::Value *val) {
  for (const auto &c : mapLookupReturns) {
    if (c.second.find(val) != c.second.end()) {
      return std::make_pair(true, mapLookupString[c.first]);
    }
  }
  return std::make_pair(false, "");
}

void ExecutionState::addBranchOnMapReturn(llvm::Value *val, const InstructionInfo *info, ref<Expr> cond) {
  BranchInfo branchInfo;
  branchInfo.sourceLine = info->line;
  branchInfo.sourceColumn = info->column;
  branchInfo.sourceFile = info->file;
  branchInfo.cond = cond;
  branchInfo.branch = val;
  
  branchesOnMapReturnReference.insert(branchInfo);
}

void ExecutionState::addMapMemoryObjects(unsigned int id, std::string allocateFunctionName) {
  MapInfo mapInfo;
  mapInfo.mapName = nextMapName;
  if (allocateFunctionName == "array_allocate") {
    mapInfo.mapType = MapType::Array;
    mapInfo.valueSize = nextValueSize;
  } else if (allocateFunctionName == "map_allocate") {
    mapInfo.mapType = MapType::Map;
    mapInfo.keySize = nextKeySize;
    mapInfo.valueSize = nextValueSize;
  } else if (allocateFunctionName == "map_of_map_allocate") {
    mapInfo.mapType = MapType::MapOfMap;
  }
  mapMemoryObjects.insert(std::make_pair(id, mapInfo));
}

MapInfo ExecutionState::getMapInfo(unsigned int id) {
  auto pos = mapMemoryObjects.find(id);
  return pos->second;
}

bool ExecutionState::isMapMemoryObject(unsigned int id) {
  return mapMemoryObjects.find(id) != mapMemoryObjects.end();
}

void ExecutionState::printMapMemoryObjects() {
  llvm::errs() << "Map Memory Objects: {\n";
  for (auto &c : mapMemoryObjects) {
    llvm::errs() << "id: " << std::to_string(c.first) 
      << ", name: " << c.second.mapName 
      << ", key size: " << std::to_string(c.second.keySize) 
      << ", value size: " << std::to_string(c.second.valueSize);
      if (c.second.mapType == MapType::Array) {
        llvm::errs() << ", mapType: Array \n";
      } else if (c.second.mapType == MapType::Map) {
        llvm::errs() << ", mapType: Map \n";
      } else if (c.second.mapType == MapType::MapOfMap) {
        llvm::errs() << ", mapType: MapOfMap \n";
      }
  }
  llvm::errs() << "}\n";
}

std::string ExecutionState::formatBranchMaps() {
  std::stringstream mapStr;

  for (auto &branch : branchesOnMapReturnReference) {
    for (auto &c: findOriginalMapCall(branch.branch)) {
      auto it = callInformation.find(c);
      if (it != callInformation.end()) {
        mapStr << " - Branch on "
               << branch.sourceFile
               << "(line:" << branch.sourceLine 
               << ", col:" << branch.sourceColumn 
               << ") used return value from "
               << it->second.functionName
               << "(" << it->second.mapName
               << ") on "
               << it->second.sourceFile
               << "(line:" << it->second.sourceLine 
               << ", col:" << it->second.sourceColumn 
               << ")";
        
        if (!branch.cond->isFalse() && !branch.cond->isTrue()) {
          mapStr << "\n   - Constraint that lead to this branch: {";
          mapStr << branch.cond;
          mapStr << "}";
        } else {
          mapStr << "\n   - Constraint on this branch not symbolic";
        }
        mapStr << "\n";
      } else {
        assert(0 && "Call to map helper function not found");
      }
    }
  }

  return mapStr.str();
}

void ExecutionState::pushFrame(KInstIterator caller, KFunction *kf) {
  stack.emplace_back(StackFrame(caller, kf));
}

void ExecutionState::popFrame() {
  const StackFrame &sf = stack.back();
  for (const auto *memoryObject : sf.allocas) {
    deallocate(memoryObject);
    addressSpace.unbindObject(memoryObject);
  }
  stack.pop_back();
}

void ExecutionState::deallocate(const MemoryObject *mo) {
  if (SingleObjectResolution) {
    auto mos_it = base_mos.find(mo->address);
    if (mos_it != base_mos.end()) {
      for (auto it = mos_it->second.begin(); it != mos_it->second.end(); ++it) {
        base_addrs.erase(*it);
      }
      base_mos.erase(mos_it->first);
    }
  }

  if (!stackAllocator || !heapAllocator)
    return;

  auto address = reinterpret_cast<void *>(mo->address);
  if (mo->isLocal) {
    stackAllocator.free(address, std::max(mo->size, mo->alignment));
  } else {
    heapAllocator.free(address, std::max(mo->size, mo->alignment));
  }
}

void ExecutionState::addSymbolic(const MemoryObject *mo, const Array *array) {
  symbolics.emplace_back(ref<const MemoryObject>(mo), array);
}

/**/

llvm::raw_ostream &klee::operator<<(llvm::raw_ostream &os, const MemoryMap &mm) {
  os << "{";
  MemoryMap::iterator it = mm.begin();
  MemoryMap::iterator ie = mm.end();
  if (it!=ie) {
    os << "MO" << it->first->id << ":" << it->second.get();
    for (++it; it!=ie; ++it)
      os << ", MO" << it->first->id << ":" << it->second.get();
  }
  os << "}";
  return os;
}

bool ExecutionState::merge(const ExecutionState &b) {
  if (DebugLogStateMerge)
    llvm::errs() << "-- attempting merge of A:" << this << " with B:" << &b
                 << "--\n";
  if (pc != b.pc)
    return false;

  // XXX is it even possible for these to differ? does it matter? probably
  // implies difference in object states?

  if (symbolics != b.symbolics)
    return false;

  {
    std::vector<StackFrame>::const_iterator itA = stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    while (itA!=stack.end() && itB!=b.stack.end()) {
      // XXX vaargs?
      if (itA->caller!=itB->caller || itA->kf!=itB->kf)
        return false;
      ++itA;
      ++itB;
    }
    if (itA!=stack.end() || itB!=b.stack.end())
      return false;
  }

  std::set< ref<Expr> > aConstraints(constraints.begin(), constraints.end());
  std::set< ref<Expr> > bConstraints(b.constraints.begin(), 
                                     b.constraints.end());
  std::set< ref<Expr> > commonConstraints, aSuffix, bSuffix;
  std::set_intersection(aConstraints.begin(), aConstraints.end(),
                        bConstraints.begin(), bConstraints.end(),
                        std::inserter(commonConstraints, commonConstraints.begin()));
  std::set_difference(aConstraints.begin(), aConstraints.end(),
                      commonConstraints.begin(), commonConstraints.end(),
                      std::inserter(aSuffix, aSuffix.end()));
  std::set_difference(bConstraints.begin(), bConstraints.end(),
                      commonConstraints.begin(), commonConstraints.end(),
                      std::inserter(bSuffix, bSuffix.end()));
  if (DebugLogStateMerge) {
    llvm::errs() << "\tconstraint prefix: [";
    for (std::set<ref<Expr> >::iterator it = commonConstraints.begin(),
                                        ie = commonConstraints.end();
         it != ie; ++it)
      llvm::errs() << *it << ", ";
    llvm::errs() << "]\n";
    llvm::errs() << "\tA suffix: [";
    for (std::set<ref<Expr> >::iterator it = aSuffix.begin(),
                                        ie = aSuffix.end();
         it != ie; ++it)
      llvm::errs() << *it << ", ";
    llvm::errs() << "]\n";
    llvm::errs() << "\tB suffix: [";
    for (std::set<ref<Expr> >::iterator it = bSuffix.begin(),
                                        ie = bSuffix.end();
         it != ie; ++it)
      llvm::errs() << *it << ", ";
    llvm::errs() << "]\n";
  }

  // We cannot merge if addresses would resolve differently in the
  // states. This means:
  // 
  // 1. Any objects created since the branch in either object must
  // have been free'd.
  //
  // 2. We cannot have free'd any pre-existing object in one state
  // and not the other

  if (DebugLogStateMerge) {
    llvm::errs() << "\tchecking object states\n";
    llvm::errs() << "A: " << addressSpace.objects << "\n";
    llvm::errs() << "B: " << b.addressSpace.objects << "\n";
  }
    
  std::set<const MemoryObject*> mutated;
  MemoryMap::iterator ai = addressSpace.objects.begin();
  MemoryMap::iterator bi = b.addressSpace.objects.begin();
  MemoryMap::iterator ae = addressSpace.objects.end();
  MemoryMap::iterator be = b.addressSpace.objects.end();
  for (; ai!=ae && bi!=be; ++ai, ++bi) {
    if (ai->first != bi->first) {
      if (DebugLogStateMerge) {
        if (ai->first < bi->first) {
          llvm::errs() << "\t\tB misses binding for: " << ai->first->id << "\n";
        } else {
          llvm::errs() << "\t\tA misses binding for: " << bi->first->id << "\n";
        }
      }
      return false;
    }
    if (ai->second.get() != bi->second.get()) {
      if (DebugLogStateMerge)
        llvm::errs() << "\t\tmutated: " << ai->first->id << "\n";
      mutated.insert(ai->first);
    }
  }
  if (ai!=ae || bi!=be) {
    if (DebugLogStateMerge)
      llvm::errs() << "\t\tmappings differ\n";
    return false;
  }
  
  // merge stack

  ref<Expr> inA = ConstantExpr::alloc(1, Expr::Bool);
  ref<Expr> inB = ConstantExpr::alloc(1, Expr::Bool);
  for (std::set< ref<Expr> >::iterator it = aSuffix.begin(), 
         ie = aSuffix.end(); it != ie; ++it)
    inA = AndExpr::create(inA, *it);
  for (std::set< ref<Expr> >::iterator it = bSuffix.begin(), 
         ie = bSuffix.end(); it != ie; ++it)
    inB = AndExpr::create(inB, *it);

  // XXX should we have a preference as to which predicate to use?
  // it seems like it can make a difference, even though logically
  // they must contradict each other and so inA => !inB

  std::vector<StackFrame>::iterator itA = stack.begin();
  std::vector<StackFrame>::const_iterator itB = b.stack.begin();
  for (; itA!=stack.end(); ++itA, ++itB) {
    StackFrame &af = *itA;
    const StackFrame &bf = *itB;
    for (unsigned i=0; i<af.kf->numRegisters; i++) {
      ref<Expr> &av = af.locals[i].value;
      const ref<Expr> &bv = bf.locals[i].value;
      if (!av || !bv) {
        // if one is null then by implication (we are at same pc)
        // we cannot reuse this local, so just ignore
      } else {
        av = SelectExpr::create(inA, av, bv);
      }
    }
  }

  for (std::set<const MemoryObject*>::iterator it = mutated.begin(), 
         ie = mutated.end(); it != ie; ++it) {
    const MemoryObject *mo = *it;
    const ObjectState *os = addressSpace.findObject(mo);
    const ObjectState *otherOS = b.addressSpace.findObject(mo);
    assert(os && !os->readOnly && 
           "objects mutated but not writable in merging state");
    assert(otherOS);

    ObjectState *wos = addressSpace.getWriteable(mo, os);
    for (unsigned i=0; i<mo->size; i++) {
      ref<Expr> av = wos->read8(i);
      ref<Expr> bv = otherOS->read8(i);
      wos->write(i, SelectExpr::create(inA, av, bv));
    }
  }

  constraints = ConstraintSet();

  ConstraintManager m(constraints);
  for (const auto &constraint : commonConstraints)
    m.addConstraint(constraint);
  m.addConstraint(OrExpr::create(inA, inB));

  return true;
}

void ExecutionState::dumpStack(llvm::raw_ostream &out) const {
  unsigned idx = 0;
  const KInstruction *target = prevPC;
  for (ExecutionState::stack_ty::const_reverse_iterator
         it = stack.rbegin(), ie = stack.rend();
       it != ie; ++it) {
    const StackFrame &sf = *it;
    Function *f = sf.kf->function;
    const InstructionInfo &ii = *target->info;
    out << "\t#" << idx++;
    std::stringstream AssStream;
    AssStream << std::setw(8) << std::setfill('0') << ii.assemblyLine;
    out << AssStream.str();
    out << " in " << f->getName().str() << "(";
    // Yawn, we could go up and print varargs if we wanted to.
    unsigned index = 0;
    for (Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
         ai != ae; ++ai) {
      if (ai!=f->arg_begin()) out << ", ";

      if (ai->hasName())
        out << ai->getName().str() << "=";

      ref<Expr> value = sf.locals[sf.kf->getArgRegister(index++)].value;
      if (isa_and_nonnull<ConstantExpr>(value)) {
        out << value;
      } else {
        out << "symbolic";
      }
    }
    out << ")";
    if (ii.file != "")
      out << " at " << ii.file << ":" << ii.line;
    out << "\n";
    target = sf.caller;
  }
}

void ExecutionState::addConstraint(ref<Expr> e) {
  ConstraintManager c(constraints);
  c.addConstraint(e);
}

void ExecutionState::addCexPreference(const ref<Expr> &cond) {
  cexPreferences = cexPreferences.insert(cond);
}
