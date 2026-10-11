#include "../src/safe_frontier.hpp"
#include <iostream>
#include <stdexcept>
using namespace tb3_frontier_exploration;
void require(bool ok,const char * why) {if (!ok) {throw std::runtime_error(why);}}
int main()
{
  std::vector<int8_t> raw(121,100), cost(121,0);
  GridView map{11,11,.1,0,0,0,raw}, costs{11,11,.1,0,0,0,cost};
  for (int x=1;x<=9;++x) {raw[map.index(x,5)]=0;}
  raw[map.index(5,5)]=-1; raw[map.index(6,5)]=-1;
  FootprintPolygon foot{{.49,.49},{.71,.49},{.71,.61},{.49,.61}};
  const auto original=raw;
  const auto reaches=[&](const FootprintPolygon * polygon,double x=.55,double y=.55) {
    return reachable_free_cells(map,costs,x,y,polygon);
  };
  auto absent=reaches(nullptr);
  require(std::none_of(absent.begin(),absent.end(),[](auto v){return v;}),"missing proof keeps refusal");
  auto result=reaches(&foot);
  require(result[map.index(9,5)] && result[map.index(1,5)],"unknown start bridge connects both known sides");
  require(raw==original,"bridge never edits the source map");
  for(size_t i=0;i<result.size();++i){require(!result[i] || raw[i]==0,"output contains known-free endpoints only");}
  raw.assign(121,-1);
  const auto island=reaches(&foot);
  require(std::none_of(island.begin(),island.end(),[](auto v){return v;}),"isolated unknown footprint cannot masquerade as mapped connectivity");
  raw=original;
  auto cw=foot; std::reverse(cw.begin(),cw.end());
  require(reaches(&cw)==result,"either polygon winding is valid");
  auto known=reaches(&foot,.45,.55);
  require(known==reaches(nullptr,.45,.55) && !known[map.index(7,5)],"known start cannot use the bridge to cross unknown");
  raw[map.index(8,5)]=-1;
  require(!reaches(&foot)[map.index(9,5)],"unknown outside footprint remains a barrier");
  raw=original; raw[map.index(6,5)]=100;
  require(!reaches(&foot)[map.index(7,5)],"occupied raw cell never traversed even with cost zero");
  raw=original; raw[map.index(5,5)]=100;
  result=reaches(&foot);
  require(std::none_of(result.begin(),result.end(),[](auto v){return v;}),"occupied actual start fails closed");
  raw=original;
  for (int value : {0,1,43,60,98}) {
    cost[map.index(5,5)]=value;
    cost[map.index(6,5)]=value;
    result=reaches(&foot);
    require(result[map.index(9,5)] && result[map.index(1,5)],
      "traversable inflation inside verified footprint must not deadlock unknown start");
    require(raw==original && !result[map.index(5,5)],
      "inflation bridge never rewrites unknown map cells or exposes them as endpoints");
    absent=reaches(nullptr);
    require(std::none_of(absent.begin(),absent.end(),[](auto v){return v;}),
      "inflated unknown start still requires verified footprint proof");
  }
  for (int value : {-1,99,100}) {
    cost[map.index(5,5)]=value;
    result=reaches(&foot);
    require(std::none_of(result.begin(),result.end(),[](auto v){return v;}),
      "unknown, inscribed or lethal actual start remains blocked");
  }
  cost.assign(121,0);
  auto thin=FootprintPolygon{{.51,.51},{.69,.51},{.69,.59},{.51,.59}};
  result=reaches(&thin);
  require(std::none_of(result.begin(),result.end(),[](auto v){return v;}),
    "center inside is insufficient: all cell corners must fit before reaching any known exit");
  for (auto bad : std::vector<FootprintPolygon>{ {},{{0,0},{1,1}},
    {{.49,.49},{.71,.61},{.71,.49},{.49,.61}},
    {{.49,.49},{.71,.49},{.60,.55},{.71,.61},{.49,.61}},
    {{.49,.49},{.60,.49},{.71,.49},{.71,.61},{.49,.61}},
    {{NAN,.49},{.71,.49},{.71,.61},{.49,.61}}, FootprintPolygon(65,{0,0})})
  {
    require(!valid_footprint(bad),"invalid footprint rejected");
    result=reaches(&bad);
    require(std::none_of(result.begin(),result.end(),[](auto v){return v;}),
      "invalid proof cannot reach any known-free exit");
  }
  // Only diagonal known-free space exists: four-connectivity cannot cut an occupied corner.
  raw.assign(121,100); raw[map.index(5,5)]=-1; raw[map.index(6,6)]=0;
  require(!reaches(&foot)[map.index(6,6)],"diagonal occupied corner remains blocked");
  raw=original;
  // Fine, rotated costmap: an off-center occupied cell is still detected.
  std::vector<int8_t> fine_data(14400,0);
  GridView fine{120,120,.025,-1,-1,.37,fine_data};
  auto fine_result=reachable_free_cells(map,fine,.55,.55,&foot);
  require(fine_result[map.index(9,5)],"different rotated cost grid supports complete coverage");
  fine_data.assign(14400,43);
  fine_result=reachable_free_cells(map,fine,.55,.55,&foot);
  require(fine_result[map.index(9,5)],"full overlapping cost coverage permits traversable inflation");
  auto [cx,cy]=fine.cell(.501,.501); fine_data[fine.index(cx,cy)]=99;
  require(fine.at(.55,.55)==43,"off-center blocker fixture leaves actual start traversable");
  fine_result=reachable_free_cells(map,fine,.55,.55,&foot);
  require(std::none_of(fine_result.begin(),fine_result.end(),[](auto v){return v;}),
    "off-center blocked overlap prevents every known-free exit, not just an unknown output mark");
  fine_data.assign(14400,0);
  // A rotated raw map has four rotated corners, not its axis-aligned envelope.
  GridView rotated{11,11,.1,.2,-.1,.43,raw};
  const auto rp=rotated.center(5,5);
  auto rotated_foot=grid_cell_corners(rotated,5,5);
  require(footprint_bridge_cell(rotated,fine,5,5,rotated_foot),"rotated map corners cover exact cell");
  require(reachable_free_cells(rotated,fine,rp.first,rp.second,&rotated_foot)[rotated.index(4,5)],
    "rotated single-cell start connects adjacent original free cell");
  // SAT ignores zero-area contact, but does not ignore a small actual overlap.
  FootprintPolygon square{{0,0},{1,0},{1,1},{0,1}};
  require(!footprint_positive_overlap(square,{{1,0},{2,0},{2,1},{1,1}}),"edge touch is not area overlap");
  require(footprint_positive_overlap(square,{{1-1e-7,0},{2,0},{2,1},{1-1e-7,1}}),"tiny positive overlap is checked");
  std::vector<int8_t> one{0}; GridView small{1,1,.08,.51,.51,0,one};
  require(!footprint_bridge_cell(map,small,5,5,foot),"uncovered costmap edge cannot be clipped away");
  // Reachability overlay does not authorize unknown final endpoints.
  result=reaches(&foot);
  auto endpoint=safe_frontier_approach(map,costs,result,{{5,5}},.55,.55);
  require(!endpoint,"unknown bridge cell is not a goal/frontier endpoint");
  endpoint=safe_frontier_approach(map,costs,result,{{7,5}},.75,.55);
  require(endpoint && map.at(endpoint->first,endpoint->second)==0,"final endpoint remains original raw free");
  std::cout << "footprint bridge safety regressions passed\n";
}
