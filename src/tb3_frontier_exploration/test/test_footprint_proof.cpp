#include "../src/footprint_proof.hpp"
#include <iostream>
#include <stdexcept>
using namespace tb3_frontier_exploration;
constexpr int64_t S=1000000000LL;
void require(bool ok,const char * why) {if(!ok){throw std::runtime_error(why);}}
geometry_msgs::msg::TransformStamped pose(int64_t ns,double x=0,double y=0,double yaw=0)
{
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id="map";t.child_frame_id="base_link";
  t.header.stamp.sec=ns/S;t.header.stamp.nanosec=ns%S;
  t.transform.translation.x=x;t.transform.translation.y=y;
  t.transform.rotation.z=std::sin(yaw/2);t.transform.rotation.w=std::cos(yaw/2);
  return t;
}
auto polygon(const geometry_msgs::msg::TransformStamped & p)
{
  auto result=std::make_shared<geometry_msgs::msg::PolygonStamped>();result->header=p.header;
  const auto t=*footprint_transform(p,"map","base_link");
  for (const auto & [x,y]:FootprintPolygon{{-.3,-.2},{.3,-.2},{.3,.2},{-.3,.2}}) {
    const auto w=t*tf2::Vector3(x,y,0);geometry_msgs::msg::Point32 v;
    v.x=w.x();v.y=w.y();v.z=w.z();result->polygon.points.push_back(v);
  }
  return result;
}
int main()
{
  const auto old=pose(10*S,2,3,.7), current=pose(11*S,-1,4,-.9);
  FootprintProofCache cache;cache.add(polygon(old),11*S);
  auto proof=cache.current("map","base_link",11*S,current,[&](int64_t ns){
    require(ns==10*S,"historical query must request original stamp");return old;
  });
  require(bool(proof),"exact original TF projects verified footprint");
  const auto expected=polygon(current);
  for(size_t i=0;i<4;++i) {
    require(std::hypot(proof->polygon[i].first-expected->polygon.points[i].x,
      proof->polygon[i].second-expected->polygon.points[i].y)<1e-6,
      "old world polygon follows current translated and rotated robot");
  }
  require(!cache.current("map","base_link",11*S,current,[&](int64_t){return current;}),
    "latest TF cannot substitute for exact historical TF");
  require(!cache.current("map","base_link",11*S,pose(10*S),[&](int64_t){return old;}),
    "stale current robot TF rejects proof");
  require(!cache.current("map","base_link",11*S,pose(11*S+1),[&](int64_t){return old;}),
    "future current robot TF rejects proof");
  // Nav2's world polygon includes full XYZ: recover the original body shape
  // with nonzero pitch/roll, then apply a different current orientation.
  auto old3=old, now3=current;
  tf2::Quaternion qo,qn;qo.setRPY(.012,.0063,.7);qn.setRPY(-.009,-.014,-.9);
  old3.transform.rotation.x=qo.x();old3.transform.rotation.y=qo.y();
  old3.transform.rotation.z=qo.z();old3.transform.rotation.w=qo.w();
  now3.transform.rotation.x=qn.x();now3.transform.rotation.y=qn.y();
  now3.transform.rotation.z=qn.z();now3.transform.rotation.w=qn.w();
  old3.transform.translation.z=.025;now3.transform.translation.z=.037;
  FootprintProofCache three_d;three_d.add(polygon(old3),11*S);
  auto proof3=three_d.current("map","base_link",11*S,now3,[&](int64_t){return old3;});
  auto expected3=polygon(now3);
  require(bool(proof3),"full 3D pose pair accepted");
  for(size_t i=0;i<4;++i) {
    require(std::hypot(proof3->polygon[i].first-expected3->polygon.points[i].x,
      proof3->polygon[i].second-expected3->polygon.points[i].y)<1e-6,
      "recover/project uses complete quaternion and XYZ, not just yaw");
  }
  // Latest sample is not in the TF cache yet; previous independently stamped sample works.
  cache.add(polygon(pose(11*S)),11*S);
  int queries=0;
  proof=cache.current("map","base_link",11*S,current,[&](int64_t ns){
    ++queries;if(ns==11*S){throw std::runtime_error("extrapolation");}return old;
  });
  require(proof && proof->source_stamp_ns==10*S && queries==2,"bounded history survives delivery phase race");
  // Freshness exact boundary and stale beyond it.
  FootprintProofCache age;age.add(polygon(old),13*S);
  require(bool(age.current("map","base_link",13*S,pose(13*S),[&](int64_t){return old;})),"three-second boundary accepted");
  require(!age.current("map","base_link",13*S+1,pose(13*S+1),[&](int64_t){return old;}),"expired proof refused");
  require(!cache.current("map","base_link",9*S,pose(9*S),[&](int64_t){return old;}),"rollback clears old epoch");
  require(!cache.current("map","base_link",11*S,current,[&](int64_t){return old;}),"old evidence never reappears as time catches up");
  for(int kind=0;kind<9;++kind) {
    FootprintProofCache bad;auto sample=polygon(old);auto historical=old;
    if(kind==0){sample->header.frame_id="odom";}
    if(kind==1){sample->header.stamp.sec=12;}
    if(kind==2){sample->polygon.points[0].z=NAN;}
    if(kind==3){sample->polygon.points.resize(2);}
    if(kind==4){historical.child_frame_id="wrong_base";}
    if(kind==5){historical.transform.rotation.w=0;historical.transform.rotation.z=0;}
    if(kind==6){for(auto & p:sample->polygon.points){p.x+=10;}}
    if(kind==7){historical.header.frame_id="odom";}
    if(kind==8){sample->header.stamp.nanosec=1000000001U;}
    bad.add(sample,11*S);
    require(!bad.current("map","base_link",11*S,current,[&](int64_t){return historical;}),
      "wrong frame/future/nonfinite/malformed/origin-detached footprint or invalid TF rejected");
  }
  FootprintProofCache bounded;
  for(int i=1;i<=4;++i){bounded.add(polygon(pose(i*S)),4*S);}
  queries=0;
  require(!bounded.current("map","base_link",4*S,pose(4*S),[&](int64_t ns){
    ++queries;require(ns!=S,"fourth old sample evicted");throw std::runtime_error("no TF");return pose(ns);
  }) && queries==3,"at most three nonblocking exact-time attempts");
  // A new callback after rollback cannot retain samples from the old epoch.
  bounded.add(polygon(pose(S)),S);
  queries=0;
  require(bool(bounded.current("map","base_link",S,pose(S),[&](int64_t ns){++queries;return pose(ns);} )) && queries==1,
    "callback rollback clears prior samples before adding new epoch footprint");
  std::cout << "footprint provenance/TF regressions passed\n";
}
