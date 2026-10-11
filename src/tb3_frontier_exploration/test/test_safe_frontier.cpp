#include "../src/safe_frontier.hpp"
#include <iostream>
#include <stdexcept>
using namespace tb3_frontier_exploration;
void require(bool ok, const char * why) {if (!ok) {throw std::runtime_error(why);}}
int main()
{
  std::vector<int8_t> cells(21*21, 0), costs_data(cells.size(), 0);
  const GridView map{21,21,.05,-.525,-.525,0,cells};
  const GridView costs{21,21,.05,-.525,-.525,0,costs_data};
  auto reachable = reachable_free_cells(map,costs,-.4,0);
  auto p = safe_frontier_approach(map,costs,reachable,{{10,10}},0,0);
  require(p && std::hypot(p->first,p->second)<1e-9,"safe cell stays at centroid");

  // The nearest candidate is to the left, on the known side of the frontier.
  for (int y=0;y<21;++y) for(int x=0;x<21;++x) {
    cells[y*21+x] = x>10 ? -1 : 0;
    costs_data[y*21+x] = x>=7 ? 85 : 40;
  }
  reachable=reachable_free_cells(map,costs,-.4,0);
  p=safe_frontier_approach(map,costs,reachable,{{10,10}},0,0);
  require(p && std::abs(p->first+.2)<1e-9 && std::abs(p->second)<1e-9,
    "retreat to nearest known-free safe cell, not unknown side");
  costs_data[10*21+11]=0;
  require(safe_frontier_approach(map,costs,reachable,{{10,10}},0,0)==p,
    "cheap unknown-side sample cannot win");

  // A cheap cell across a wall is not connected to the frontier side.
  cells.assign(cells.size(),0); costs_data.assign(costs_data.size(),85);
  for(int y=0;y<21;++y) cells[y*21+9]=100;
  costs_data[10*21+8]=0;
  reachable=reachable_free_cells(map,costs,.2,0);
  require(!safe_frontier_approach(map,costs,reachable,{{10,10}},0,0),
    "cannot jump across occupied wall to cheap cell");
  for(int y=0;y<21;++y) cells[y*21+9]=-1;
  require(!safe_frontier_approach(map,costs,reachable,{{10,10}},0,0),
    "cannot jump across unknown strip");

  // Robot and frontier are disconnected even though a local cheap point exists.
  cells.assign(cells.size(),0); costs_data.assign(costs_data.size(),0);
  for(int y=0;y<21;++y) cells[y*21+9]=100;
  reachable=reachable_free_cells(map,costs,-.3,0);
  require(!safe_frontier_approach(map,costs,reachable,{{10,10}},0,0),
    "local safe component is not reachable from robot");
  cells.assign(cells.size(),100); costs_data.assign(costs_data.size(),0);
  cells[10*21+10]=cells[11*21+11]=0;
  reachable=reachable_free_cells(map,costs,0,0);
  require(!reachable[11*21+11],"diagonal occupied corner cannot be cut");

  // A 0.55m slot has at most .275m clearance. Approximate Nav2 inflation:
  // radius .22, scaling 3 => grid cost ~84 at center, not a safe endpoint.
  cells.assign(cells.size(),0); costs_data.assign(costs_data.size(),0);
  for(int y=0;y<21;++y) for(int x=0;x<21;++x) {
    const auto [wx,wy]=map.center(x,y);
    if(wy>=-.25) {
      const double clearance=.275-std::abs(wx);
      if(clearance<=0) {cells[y*21+x]=100; costs_data[y*21+x]=100;}
      else costs_data[y*21+x]=clearance<=.22 ? 99 :
        static_cast<int8_t>(std::round(98*std::exp(-3*(clearance-.22))));
    }
  }
  reachable=reachable_free_cells(map,costs,0,-.4);
  p=safe_frontier_approach(map,costs,reachable,{{10,10}},0,0);
  require(p && p->second<-.25 && costs.at(p->first,p->second)<=60,
    "narrow slot frontier retreats to its safe entrance");
  require(!safe_frontier_approach(map,costs,reachable,{{10,18}},0,.4),
    "deep slot with no safe point within .5m is rejected");

  cells.assign(cells.size(),0); costs_data.assign(costs_data.size(),85);
  costs_data[10*21+0]=60;
  reachable=reachable_free_cells(map,costs,0,0);
  require(safe_frontier_approach(map,costs,reachable,{{10,10}},0,0,.5).has_value(),
    "radius boundary .5m is inclusive");
  require(!safe_frontier_approach(map,costs,reachable,{{10,10}},0,0,.49),
    "point outside search radius rejected");
  require(!safe_frontier_approach(map,costs,reachable,{{10,10}},0,0,.5,60,1),
    "costmap edge margin applies to shifted point");
  costs_data[10*21+0]=-1;
  require(!safe_frontier_approach(map,costs,reachable,{{10,10}},0,0),
    "unknown costmap value cannot pass threshold");
  std::vector<int8_t> malformed(1,0);
  const GridView invalid{21,21,.05,0,0,0,malformed};
  require(reachable_free_cells(map,invalid,0,0).empty(),"malformed costmap fails closed");

  // Origin and resolution are handled independently, including rotated grids.
  costs_data.assign(costs_data.size(),0);
  const GridView rotated{21,21,.05,.525,-.525,std::acos(-1.0)/2,costs_data};
  require(rotated.at(.2,-.1)==0,"rotated costmap lookup");
  reachable=reachable_free_cells(map,rotated,0,0);
  require(safe_frontier_approach(map,rotated,reachable,{{10,10}},0,0).has_value(),
    "rotated costmap preserves approach eligibility");
  std::vector<int8_t> fine_data(42*42,0);
  const GridView fine{42,42,.025,-.525,-.525,0,fine_data};
  reachable=reachable_free_cells(map,fine,0,0);
  require(safe_frontier_approach(map,fine,reachable,{{10,10}},0,0).has_value(),
    "different grid resolutions preserve geometry");

  // Broad closed rings and U-shaped frontier clusters need not have any
  // frontier cell near their arithmetic centre. Preserve a useful boundary
  // goal instead of retreating unnecessarily toward the robot at the centre.
  for (int half : {15, 30}) for (bool closed : {true, false}) {
    const int n = 2*half+5, center = half+2;
    std::vector<int8_t> region(n*n, -1), band_costs(n*n, 0);
    const GridView broad{n,n,.05,-(center+.5)*.05,-(center+.5)*.05,0,region};
    const GridView broad_costs{n,n,.05,broad.x,broad.y,0,band_costs};
    std::vector<GridCell> boundary;
    double cx=0, cy=0;
    for (int y=2;y<n-2;++y) for (int x=2;x<n-2;++x) region[y*n+x]=0;
    if (!closed) {  // Open the top into known space; the remaining edge is U-shaped.
      for(int y=n-2;y<n;++y) for(int x=0;x<n;++x) region[y*n+x]=0;
    }
    // Derive actual frontier cells with the node's free/unknown adjacency rule.
    for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
      if(region[y*n+x]!=0) continue;
      bool frontier=false;
      for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int nx=x+dx,ny=y+dy;
        if(nx>=0 && ny>=0 && nx<n && ny<n && region[ny*n+nx]<0) frontier=true;
      }
      if(frontier) {
        boundary.emplace_back(x,y);
        const auto [wx,wy]=broad.center(x,y); cx+=wx; cy+=wy;
      }
    }
    cx/=boundary.size(); cy/=boundary.size();
    const auto broad_reached=reachable_free_cells(broad,broad_costs,0,0);
    GridCell selected;
    auto choice=safe_frontier_approach(broad,broad_costs,broad_reached,boundary,cx,cy,.5,60,0,&selected);
    require(choice.has_value(),"ring/U boundary remains usable at either scale");
    const auto cell=broad.cell(choice->first,choice->second);
    require(std::find(boundary.begin(),boundary.end(),cell)!=boundary.end(),
      "safe frontier needs zero retreat, not a centroid-nearest interior point");
    require(std::hypot(choice->first,choice->second)>.5,
      "broad frontier candidate survives minimum navigation distance");
    require(selected==cell,"zero-retreat choice retains its actual frontier seed");
    const auto yaw=frontier_outward_yaw(broad,selected,choice->first,choice->second);
    require(yaw && broad.at(choice->first+.05*std::cos(*yaw),
                           choice->second+.05*std::sin(*yaw))<0,
      "ring/U camera faces adjacent unknown space, not the inside centroid");
  }

  // Search may retreat from an inflated/lethal-cost frontier into safe free
  // space: the endpoint, not this hypothetical retreat, is the robot's goal.
  cells.assign(cells.size(),0); costs_data.assign(costs_data.size(),40);
  for(int y=0;y<21;++y) for(int x=0;x<21;++x) {
    if(x>10) cells[y*21+x]=-1;
    if(x>=7) costs_data[y*21+x]=100;
  }
  reachable=reachable_free_cells(map,costs,-.4,0);
  GridCell selected;
  p=safe_frontier_approach(map,costs,reachable,{{10,10}},0,0,.5,60,0,&selected);
  require(p && std::abs(p->first+.2)<1e-9 && std::abs(p->second)<1e-9,
    "high-cost boundary can retreat to an independently reachable safe endpoint");
  require(selected==GridCell{10,10},"retreated choice retains actual boundary seed");
  auto yaw=frontier_outward_yaw(map,selected,p->first,p->second);
  require(yaw && std::abs(*yaw)<1e-9,"wall boundary faces east into unknown after safe retreat");
  const GridView turned{21,21,.05,.525,-.525,std::acos(-1.0)/2,cells};
  const auto turned_goal=turned.center(6,10);
  yaw=frontier_outward_yaw(turned,selected,turned_goal.first,turned_goal.second);
  require(yaw && std::abs(*yaw-std::acos(-1.0)/2)<1e-9,
    "boundary heading respects rotated map origin");

  cells.assign(cells.size(),100); cells[10*21+10]=0;
  cells[10*21+9]=cells[10*21+11]=-1;
  yaw=frontier_outward_yaw(map,{10,10},0,0);
  require(yaw && map.at(.05*std::cos(*yaw),.05*std::sin(*yaw))<0,
    "opposite unknown directions fall back to an actual unknown neighbour");
  cells[10*21+9]=cells[10*21+11]=0;
  require(!frontier_outward_yaw(map,{10,10},0,0),
    "non-frontier point cannot invent a camera heading");
  cells.assign(cells.size(),100); cells[10*21+10]=0;
  cells[11*21+9]=cells[11*21+11]=-1;
  yaw=frontier_outward_yaw(map,{10,10},0,0);
  require(yaw && map.at(.05*std::cos(*yaw),.05*std::sin(*yaw))<0,
    "unknown-neighbour average must not point at the occupied cell between them");

  // Retreat uses four-connected path length, not a Euclidean disc that can
  // hide an arbitrarily long detour behind a wall or unknown strip.
  cells.assign(cells.size(),100); costs_data.assign(costs_data.size(),85);
  for(int y=6;y<=11;++y) {cells[y*21+6]=0; cells[y*21+8]=0;}
  cells[11*21+7]=0;
  costs_data[8*21+8]=0;
  const auto seed=map.center(6,6), safe=map.center(8,8);
  reachable=reachable_free_cells(map,costs,safe.first,safe.second);
  require(!safe_frontier_approach(map,costs,reachable,{{6,6}},seed.first,seed.second,.499),
    "ten-step .5m detour cannot fit below the retreat budget");
  p=safe_frontier_approach(map,costs,reachable,{{6,6}},seed.first,seed.second,.5);
  require(p && std::hypot(p->first-safe.first,p->second-safe.second)<1e-9,
    "ten-step .5m retreat budget is inclusive");
  costs_data[8*21+8]=85; costs_data[6*21+8]=0;
  const auto farther=map.center(8,6);
  reachable=reachable_free_cells(map,costs,farther.first,farther.second);
  require(!safe_frontier_approach(map,costs,reachable,{{6,6}},seed.first,seed.second,.5),
    ".1m Euclidean neighbour behind a .6m detour exceeds retreat budget");
  for(int y=6;y<11;++y) cells[y*21+7]=-1;
  require(!safe_frontier_approach(map,costs,reachable,{{6,6}},seed.first,seed.second,.5),
    "unknown shortcut cannot reduce the four-connected retreat distance");
  std::cout<<"safe frontier regressions passed\n";
}
