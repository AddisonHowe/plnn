#!/usr/bin/env wolframscript
(* ::Package:: *)

X10=ToExpression[$ScriptCommandLine[[2]]];
X20=ToExpression[$ScriptCommandLine[[3]]];
S1=ToExpression[$ScriptCommandLine[[4]]];
S2=ToExpression[$ScriptCommandLine[[5]]];
dt=ToExpression[$ScriptCommandLine[[6]]];


signal={{S1}, {S2}};


(* Starting Weights *)
(* W1={{1,3},{2,2},{3,1}}; *)
(* W2={{1,1,-2},{0,1,0},{-1,2,1}}; *)
(* W3={{2,3,1}}; *)
(* WT={{2,4},{-1,1}}; *)


(* Weights after update from training over ALL data *)
W1={
   {1.00795443, 2.99556809},
   {1.99612237, 1.97785077},
   {3.01195923, 0.99704222}
};
W2={
   { 0.93806393,  0.94526547, -2.05927363},
   {-0.05160056,  0.95660619, -0.03341331},
   {-1.00795646,  1.99330224,  0.99321868}
};
W3={
   {2.00279667, 2.98271836, 0.97964255}
};
WT={
   { 2.03202278,  3.97701852},
   {-0.99596133,  0.98111955}
};


act[x_]:=Tanh[x];


f1[x1_,x2_]:=act[Dot[W1,{{x1}, {x2}}]];
f2[y_]:=act[Dot[W2,y]];
f3[y_]:=Dot[W3,y];
Phi[x1_, x2_]:=f3[f2[f1[x1, x2]]][[1,1]];


X1=X10;
X2=X20;
(*Print["signal: ", signal]*)
(*Print["Phi: ",N[Phi[X1, X2]]]*)
dphidx=D[Phi[x1, x2], {x1, 1}]/.{x1->X1,x2->X2};
dphidy=D[Phi[x1, x2], {x2, 1}]/.{x1->X1,x2->X2};
(*Print["grad Phi: ", {N[dphidx], N[dphidy]}]*)
tilt = Dot[WT, signal];
(*Print["tilt Tau: ", tilt]*)
f=-({{dphidx}, {dphidy}} + tilt);
(*Print["F: ", N[f]]*)
dX=f*dt;
(*Print["dX: ", dX]*)
newX1=X1+dX[[1]];
newX2=X2+dX[[2]];
Print[Flatten[{newX1,newX2}]]


X1=newX1;
X2=newX2;
(*Print["signal: ", signal]*)
(*Print["Phi: ",N[Phi[X1, X2]]]*)
dphidx=D[Phi[x1, x2], {x1, 1}]/.{x1->X1,x2->X2};
dphidy=D[Phi[x1, x2], {x2, 1}]/.{x1->X1,x2->X2};
(*Print["grad Phi: ", {N[dphidx], N[dphidy]}]*)
tilt = Dot[WT, signal];
(*Print["tilt Tau: ", tilt]*)
f=-({{dphidx}, {dphidy}} + tilt);
(*Print["F: ", N[f]]*)
dX=f*dt;
(*Print["dX: ", dX]*)
newX1=X1+dX[[1]];
newX2=X2+dX[[2]];
(*Print[Flatten[{newX1,newX2}]]*)
