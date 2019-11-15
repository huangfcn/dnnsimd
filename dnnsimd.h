#ifndef __DNNLIB_SIMD_C__
#define __DNNLIB_SIMD_C__

#ifdef __cplusplus
extern "C" {
#endif

cnn_type_t macpool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc
);

void macpool4(
   const int          _n,
   const int          _wsiz,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _zloc
);

void macpool8(
   const int          _n,
   const int          _wsiz,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _zloc
);

void addpool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _yloc
);

void subpool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _yloc
);

void mulpool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _yloc
);

void addpool2(
         int          _n,
   const cnn_type_t * _xloc,
         cnn_type_t   _wloc,
         cnn_type_t * _yloc
);

void mulpool2(
         int          _n,
   const cnn_type_t * _xloc,
         cnn_type_t   _wloc,
         cnn_type_t * _yloc
);

void conv3x3(
         int            M,
         int            N,
         int            C1,
         int            C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv2x2(
         int            M,
         int            N,
         int            C1,
         int            C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv1x1(
         int            M,
         int            N,
         int            C1,
         int            C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv3x3_s2(
         int            M,
         int            N,
         int            C1,
         int            C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv3x3_p1(
         int            M,
         int            N,
         int            C1,
         int            C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv3x3_p1s2_bnrelu_inp(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
   const cnn_type_t * __pu,
   const cnn_type_t * __ps,
   const cnn_type_t * __pb,
   const cnn_type_t   maxd,
         cnn_type_t * __py
);

void dw_conv3x3_p1s1_bnrelu(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
   const cnn_type_t * __pu,
   const cnn_type_t * __ps,
   const cnn_type_t * __pb,
   const cnn_type_t   maxd,
         cnn_type_t * __py
);

void dw_conv3x3_p1s2_bnrelu(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
   const cnn_type_t * __pu,
   const cnn_type_t * __ps,
   const cnn_type_t * __pb,
   const cnn_type_t   maxd,
         cnn_type_t * __py
);

void conv1x1_p0s1_bn(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
   const cnn_type_t * __pu,
   const cnn_type_t * __ps,
   const cnn_type_t * __pb,
         cnn_type_t * __py
);

void conv1x1_p0s1_bnrelu(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
   const cnn_type_t * __pu,
   const cnn_type_t * __ps,
   const cnn_type_t * __pb,
   const cnn_type_t   maxd,
         cnn_type_t * __py
);

void conv1x3_p1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv3x1_p1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv1x7_p3(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv7x1_p3(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void conv5x5_p2(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * __px,
   const cnn_type_t * __pw,
         cnn_type_t * __py
);

void fullnet(
         int          M,
         int          N,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _yloc   
);

/* maxpool3x3 - full */
void maxpool3x3(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

/* maxpool3x3 - valid */
void maxpool3x3_(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

void maxpool2x2(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

void avgpool2x2(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

void avgpool3x3_p1(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * __px,
         cnn_type_t * __py
);

/* avgpool3x3 - valid */
void avgpool3x3_(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

void bnormpool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _uloc,
   const cnn_type_t * _varn,
   const cnn_type_t * _beta,
         cnn_type_t * _yloc
);

void bnrelupool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _uloc,
   const cnn_type_t * _varn,
   const cnn_type_t * _beta,
         cnn_type_t * _yloc
);

void b35respool(
         int          _n,
   const cnn_type_t * _z0loc,
   const cnn_type_t * _z1loc,
   const cnn_type_t * _z2loc,
   const cnn_type_t * _bloc,
         cnn_type_t   _scl,
   const cnn_type_t * _xloc,
         cnn_type_t * _yloc
);

void b17respool(
         int          _n,
   const cnn_type_t * _z0loc,
   const cnn_type_t * _z1loc,
   const cnn_type_t * _bloc,
         cnn_type_t   _scl,
   const cnn_type_t * _xloc,
         cnn_type_t * _yloc
);

void b08respool(
         int          _n,
   const cnn_type_t * _z0loc,
   const cnn_type_t * _z1loc,
   const cnn_type_t * _bloc,
         cnn_type_t   _scl,
   const cnn_type_t * _xloc,
         cnn_type_t * _yloc
);

void b08respool2(
         int          _n,
   const cnn_type_t * _z0loc,
   const cnn_type_t * _z1loc,
   const cnn_type_t * _bloc,
         cnn_type_t   _scl,
   const cnn_type_t * _xloc,
         cnn_type_t * _yloc
);


void b01respool(
         int          _n,
   const cnn_type_t * _zloc,
   const cnn_type_t * _bloc,
         cnn_type_t   _scl,
   const cnn_type_t * _xloc,
         cnn_type_t * _yloc
);


void b01respool2(
         int          _n,
   const cnn_type_t * _zloc,
   const cnn_type_t * _bloc,
         cnn_type_t   _scl,
   const cnn_type_t * _xloc,
         cnn_type_t * _yloc
);

#define b02respool b08respool
#define b03respool b35respool

void conv2d_concat(
         int  M,
         int  N,
         int  Cx,
         int  Cy,
         int  Sy,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

#ifdef __cplusplus
};
#endif

#endif