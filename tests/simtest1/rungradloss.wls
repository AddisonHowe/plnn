#!/usr/bin/env wolframscript
(* ::Package:: *)

X1=ToExpression[$ScriptCommandLine[[2]]];
Y1=ToExpression[$ScriptCommandLine[[3]]];
X2=ToExpression[$ScriptCommandLine[[4]]];
Y2=ToExpression[$ScriptCommandLine[[5]]];
X3=ToExpression[$ScriptCommandLine[[6]]];
Y3=ToExpression[$ScriptCommandLine[[7]]];
X4=ToExpression[$ScriptCommandLine[[8]]];
Y4=ToExpression[$ScriptCommandLine[[9]]];
S1=ToExpression[$ScriptCommandLine[[10]]];
S2=ToExpression[$ScriptCommandLine[[11]]];
dt=ToExpression[$ScriptCommandLine[[12]]];


W1[w111_,w121_,w211_,w221_,w311_,w321_]:={
{w111,w121},
{w211,w221},
{w311,w321}
};
W2[w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_]:={
{w112,w122,w132},
{w212,w222,w232},
{w312,w322,w332}
};
W3[w113_,w123_,w133_]:={
{w113,w123,w133}
};
WT[ws11_,ws12_,ws21_,ws22_]:={
{ws11,ws12},
{ws21,ws22}
};
act[x_]:=Tanh[x];


W1sym={{w111,w121},{w211,w221},{w311,w321}};
W2sym={{w112,w122,w132},{w212,w222,w232},{w312,w322,w332}};
W3sym={{w113,w123,w133}};
WTsym={{ws11,ws12},{ws21,ws22}};


(* Starting Weights *)
(* W1star={{1,3},{2,2},{3,1}}; *)
(* W2star={{1,1,-2},{0,1,0},{-1,2,1}}; *)
(* W3star={{2,3,1}}; *)
(* WTstar={{2,4},{-1,1}}; *)


(* Weights after update from training over ALL data *)
W1star={
   {1.00795443, 2.99556809},
   {1.99612237, 1.97785077},
   {3.01195923, 0.99704222}
};
W2star={
   { 0.93806393,  0.94526547, -2.05927363},
   {-0.05160056,  0.95660619, -0.03341331},
   {-1.00795646,  1.99330224,  0.99321868}
};
W3star={
   {2.00279667, 2.98271836, 0.97964255}
};
WTstar={
   { 2.03202278,  3.97701852},
   {-0.99596133,  0.98111955}
};


subs={
w111->W1star[[1,1]],
w121->W1star[[1,2]],
w211->W1star[[2,1]],
w221->W1star[[2,2]],
w311->W1star[[3,1]],
w321->W1star[[3,2]],
w112->W2star[[1,1]],
w122->W2star[[1,2]],
w132->W2star[[1,3]],
w212->W2star[[2,1]],
w222->W2star[[2,2]],
w232->W2star[[2,3]],
w312->W2star[[3,1]],
w322->W2star[[3,2]],
w332->W2star[[3,3]],
w113->W3star[[1,1]],
w123->W3star[[1,2]],
w133->W3star[[1,3]],
ws11->WTstar[[1,1]],
ws12->WTstar[[1,2]],
ws21->WTstar[[2,1]],
ws22->WTstar[[2,2]]
};


f1[x1_,x2_,w111_,w121_,w211_,w221_,w311_,w321_]:=
act[Dot[
W1[w111,w121,w211,w221,w311,w321],
{{x1}, {x2}}
]];

f2[y_, w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_]:=
act[Dot[
W2[w112,w122,w132,w212,w222,w232,w312,w322,w332],
y
]];

f3[y_, w113_,w123_,w133_]:=
Dot[
W3[w113,w123,w133],
y
];

Phi[x1_, x2_, 
w111_,w121_,w211_,w221_,w311_,w321_,
w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_,
w113_,w123_,w133_]:=
f3[
f2[
f1[x1, x2,w111,w121,w211,w221,w311,w321],
w112,w122,w132,w212,w222,w232,w312,w322,w332
],
w113,w123,w133
][[1,1]];

Tau[s1_,s2_,ws11_,ws12_,ws21_,ws22_]:=Dot[WT[ws11,ws12,ws21,ws22],{{s1},{s2}}]

dphidx[
xv1_,xv2_,
w111_,w121_,w211_,w221_,w311_,w321_,
w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_,
w113_,w123_,w133_]:=D[Phi[x1, x2,w111,w121,w211,w221,w311,w321,w112,w122,w132,w212,w222,w232,w312,w322,w332,w113,w123,w133], {x1, 1}]/.{x1->xv1,x2->xv2};

dphidy[
xv1_,xv2_,
w111_,w121_,w211_,w221_,w311_,w321_,
w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_,
w113_,w123_,w133_]:=D[Phi[x1, x2,w111,w121,w211,w221,w311,w321,w112,w122,w132,w212,w222,w232,w312,w322,w332,w113,w123,w133], {x2, 1}]/.{x1->xv1,x2->xv2};


evolve[
x0_,y0_,s1_,s2_,
w111_,w121_,w211_,w221_,w311_,w321_,
w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_,
w113_,w123_,w133_,
ws11_,ws12_,ws21_,ws22_
]:={{x0-dt*(dphidx[x0,y0,w111,w121,w211,w221,w311,w321,w112,w122,w132,w212,w222,w232,w312,w322,w332,w113,w123,w133] + Tau[s1,s2,ws11,ws12,ws21,ws22][[1]])}, {y0-dt*(dphidy[x0,y0,w111,w121,w211,w221,w311,w321,w112,w122,w132,w212,w222,w232,w312,w322,w332,w113,w123,w133]+Tau[s1,s2,ws11,ws12,ws21,ws22][[2]])}}


evolve2[
x0_,y0_,s1_,s2_,
w111_,w121_,w211_,w221_,w311_,w321_,
w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_,
w113_,w123_,w133_,
ws11_,ws12_,ws21_,ws22_
]:=evolve[
evolve[
x0,y0,s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22
][[1,1]],
evolve[
x0,y0,s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22
][[2,1]],
s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22]


Loss[
x10_,y10_,x20_,y20_,x30_,y30_,x40_,y40_,
w111_,w121_,w211_,w221_,w311_,w321_,
w112_,w122_,w132_,w212_,w222_,w232_,w312_,w322_,w332_,
w113_,w123_,w133_,
ws11_,ws12_,ws21_,ws22_,
s1_,s2_] := Total[
   (0.25Flatten[
evolve2[
x10,y10,s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22
]+
evolve2[
x20,y20,s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22
]+
evolve2[
x30,y30,s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22
]+
evolve2[
x40,y40,s1,s2,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22
]] - {0.5, 0.5})^2
]


fgradw=Function[w,
D[Loss[
X1,Y1,X2,Y2,X3,Y3,X4,Y4,
w111,w121,w211,w221,w311,w321,
w112,w122,w132,w212,w222,w232,w312,w322,w332,
w113,w123,w133,
ws11,ws12,ws21,ws22,
S1,S2], {w,1}]/.subs
];


Print[Map[fgradw, W1sym]]
Print[Map[fgradw, W2sym]]
Print[Map[fgradw, W3sym]]
Print[Map[fgradw, WTsym]]
