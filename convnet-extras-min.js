var A=a=>typeof a=='undefined';(function(_){"use strict";var b=function(C){var C=C||{};this.alpha=C.alpha||0.01;this.out_sx=C.in_sx;this.out_sy=C.in_sy;this.out_depth=C.in_depth;this.layer_type='leaky_relu'},c=function(E){var E=E||{};this.alpha=E.alpha||0.01;this.out_sx=E.in_sx;this.out_sy=E.in_sy;this.out_depth=E.in_depth;this.layer_type='elu'},d=function(aE){var aE=aE||{};this.b=aE.b||0.05;this.out_sx=aE.in_sx;this.out_sy=aE.in_sy;this.out_depth=aE.in_depth;this.layer_type='frelu'},e=function(aJ){var aJ=aJ||{};this.out_sx=aJ.in_sx;this.out_sy=aJ.in_sy;this.out_depth=aJ.in_depth;this.layer_type='swish'},f=function(aP){var aP=aP||{};this.alpha=aP.alpha||0.01;this.out_sx=aP.in_sx;this.out_sy=aP.in_sy;this.out_depth=aP.in_depth;this.layer_type='plu'},g=function(aV){var aV=aV||{};this.alpha=aV.alpha||0.5;this.out_sx=aV.in_sx;this.out_sy=aV.in_sy;this.out_depth=aV.in_depth;this.layer_type='double_relu'},h=function(bB){var bB=bB||{};this.alpha=bB.alpha||1.5;this.beta=bB.beta||3;this.gamma=bB.gamma||1;this.out_sx=bB.in_sx;this.out_sy=bB.in_sy;this.out_depth=bB.in_depth;this.layer_type='pilu'},B=function(bI){var bI=bI||{};this.out_sx=bI.in_sx;this.out_sy=bI.in_sy;this.out_depth=bI.in_depth;this.layer_type='mish'},J=function(bP){var bP=bP||{};this.out_sx=bP.in_sx;this.out_sy=bP.in_sy;this.out_depth=bP.in_depth;this.layer_type='smish'},k=function(bV){var bV=bV||{};this.out_sx=bV.in_sx;this.out_sy=bV.in_sy;this.out_depth=bV.in_depth;this.layer_type='softplus'},m=function(cC){var cC=cC||{};this.out_sx=cC.in_sx;this.out_sy=cC.in_sy;this.out_depth=cC.in_depth;this.layer_type='tanh'};b.prototype={forward:function(V){this.in_act=V;var _b=V.clone(),N=V.w.length,D=_b.w;for(var i=0;i<N;i++)D[i]<0&&(D[i]=D[i]*this.alpha);this.out_act=_b;return this.out_act},backward:function(){var V=this.in_act,_B=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++)_B.w[i]<=0?V.dw[i]=_B.dw[i]*this.alpha:V.dw[i]=_B.dw[i]},getParamsAndGrads:function(){return[]},toJSON:function(){var _a={};_a.out_depth=this.out_depth;_a.out_sx=this.out_sx;_a.out_sy=this.out_sy;_a.layer_type=this.layer_type;return _a},fromJSON:function(_A){this.out_depth=_A.out_depth;this.out_sx=_A.out_sx;this.out_sy=_A.out_sy;this.layer_type=_A.layer_type}};c.prototype={forward:function(V){this.in_act=V;var aA=V.clone(),N=V.w.length,_d=aA.w;for(var i=0;i<N;i++)_d[i]<=0&&(_d[i]=this.alpha*(Math.exp(_d[i])-1));this.out_act=aA;return this.out_act},backward:function(){var V=this.in_act,aB=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++)aB.w[i]<=0?V.dw[i]=aB.dw[i]*Math.exp(aB.dw[i])*this.alpha:V.dw[i]=aB.dw[i]},getParamsAndGrads:function(){return[]},toJSON:function(){var aC={};aC.out_depth=this.out_depth;aC.out_sx=this.out_sx;aC.out_sy=this.out_sy;aC.layer_type=this.layer_type;return aC},fromJSON:function(aD){this.out_depth=aD.out_depth;this.out_sx=aD.out_sx;this.out_sy=aD.out_sy;this.layer_type=aD.layer_type}};d.prototype={forward:function(V){this.in_act=V;var aF=V.clone(),N=V.w.length,_D=aF.w;for(var i=0;i<N;i++)_D[i]<=0?_D[i]=this.b:_D[i]=_D[i]+this.b;this.out_act=aF;return this.out_act},backward:function(){var V=this.in_act,aG=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++)aG.w[i]<=0?V.dw[i]=0:V.dw[i]=aG.dw[i]},getParamsAndGrads:function(){return[]},toJSON:function(){var aH={};aH.out_depth=this.out_depth;aH.out_sx=this.out_sx;aH.out_sy=this.out_sy;aH.layer_type=this.layer_type;return aH},fromJSON:function(aI){this.out_depth=aI.out_depth;this.out_sx=aI.out_sx;this.out_sy=aI.out_sy;this.layer_type=aI.layer_type}};e.prototype={forward:function(V){this.in_act=V;var aK=V.clone(),N=V.w.length,aL=aK.w,_e=V.w;for(var i=0;i<N;i++)aL[i]=_e[i]/(1.0+Math.exp(-_e[i]));this.out_act=aK;return this.out_act},backward:function(){var V=this.in_act,aM=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++){var _E=aM.w[i];V.dw[i]=(V.w[i]*(_E-_E*_E)+_E)*aM.dw[i]}},getParamsAndGrads:function(){return[]},toJSON:function(){var aN={};aN.out_depth=this.out_depth;aN.out_sx=this.out_sx;aN.out_sy=this.out_sy;aN.layer_type=this.layer_type;return aN},fromJSON:function(aO){this.out_depth=aO.out_depth;this.out_sx=aO.out_sx;this.out_sy=aO.out_sy;this.layer_type=aO.layer_type}};f.prototype={forward:function(V){this.in_act=V;var aQ=V.clone(),N=V.w.length,aR=aQ.w;for(var i=0;i<N;i++)aR[i]>1?aR[i]=(aR[i]-1)*this.alpha+1:aR[i]<-1&&(aR[i]=(aR[i]+1)*this.alpha-1);this.out_act=aQ;return this.out_act},backward:function(){var V=this.in_act,aS=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++)aS.w[i]<-1||aS.w[i]>1?V.dw[i]=aS.dw[i]*this.alpha:V.dw[i]=aS.dw[i]},getParamsAndGrads:function(){return[]},toJSON:function(){var aT={};aT.out_depth=this.out_depth;aT.out_sx=this.out_sx;aT.out_sy=this.out_sy;aT.layer_type=this.layer_type;return aT},fromJSON:function(aU){this.out_depth=aU.out_depth;this.out_sx=aU.out_sx;this.out_sy=aU.out_sy;this.layer_type=aU.layer_type}};g.prototype={forward:function(V){this.in_act=V;var aW=V.clone(),N=V.w.length,aX=aW.w;for(var i=0;i<N;i++)aX[i]<this.alpha?aX[i]=aX[i]+this.alpha:aX[i]>this.alpha&&(aX[i]=aX[i]-this.alpha);this.out_act=aW;return this.out_act},backward:function(){var V=this.in_act,aY=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++)Math.abs(aY.w[i])>this.alpha?V.dw[i]=aY.dw[i]*this.alpha:V.dw[i]=aY.dw[i]},getParamsAndGrads:function(){return[]},toJSON:function(){var aZ={};aZ.out_depth=this.out_depth;aZ.out_sx=this.out_sx;aZ.out_sy=this.out_sy;aZ.layer_type=this.layer_type;return aZ},fromJSON:function(bA){this.out_depth=bA.out_depth;this.out_sx=bA.out_sx;this.out_sy=bA.out_sy;this.layer_type=bA.layer_type}};h.prototype={forward:function(V){this.in_act=V;var bC=V.clone(),N=V.w.length,bD=bC.w,bE=V.w;for(var i=0;i<N;i++)bE[i]>gamma?bD[i]=alpha*bE[i]+gamma*(1-alpha):bD[i]=beta*bE[i]+gamma*(1-beta);this.out_act=bC;return this.out_act},backward:function(){var V=this.in_act,bF=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++)V.w[i]>gamma?V.dw[i]=alpha*bF.dw[i]:V.dw[i]=beta*bF.dw[i]},getParamsAndGrads:function(){return[]},toJSON:function(){var bG={};bG.out_depth=this.out_depth;bG.out_sx=this.out_sx;bG.out_sy=this.out_sy;bG.layer_type=this.layer_type;return bG},fromJSON:function(bH){this.out_depth=bH.out_depth;this.out_sx=bH.out_sx;this.out_sy=bH.out_sy;this.layer_type=bH.layer_type}};B.prototype={forward:function(V){this.in_act=V;var bJ=V.clone(),N=V.w.length,bK=bJ.w,bL=V.w;for(var i=0;i<N;i++)bK[i]=bL[i]*Math.tanh(Math.log(1+Math.exp(bL[i])));this.out_act=bJ;return this.out_act},backward:function(){var V=this.in_act,bM=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++){var x=V.w[i],F=Math.log(1+Math.exp(x)),G=Math.tanh(F),H=1-Math.pow(G,2),I=1/(1+Math.exp(-x));V.dw[i]=bM.dw[i]*(G+x*H*I)}},getParamsAndGrads:function(){return[]},toJSON:function(){var bN={};bN.out_depth=this.out_depth;bN.out_sx=this.out_sx;bN.out_sy=this.out_sy;bN.layer_type=this.layer_type;return bN},fromJSON:function(bO){this.out_depth=bO.out_depth;this.out_sx=bO.out_sx;this.out_sy=bO.out_sy;this.layer_type=bO.layer_type}};J.prototype={forward:function(V){this.in_act=V;var bQ=V.clone(),N=V.w.length,bR=bQ.w,bS=V.w;for(var i=0;i<N;i++){var _g=1/(1+Math.exp(-bS[i]));bR[i]=bS[i]*Math.tanh(_g)}this.out_act=bQ;return this.out_act},backward:function(){var V=this.in_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++){}},getParamsAndGrads:function(){return[]},toJSON:function(){var bT={};bT.out_depth=this.out_depth;bT.out_sx=this.out_sx;bT.out_sy=this.out_sy;bT.layer_type=this.layer_type;return bT},fromJSON:function(bU){this.out_depth=bU.out_depth;this.out_sx=bU.out_sx;this.out_sy=bU.out_sy;this.layer_type=bU.layer_type}};k.prototype={forward:function(V){this.in_act=V;var bW=V.clone(),N=V.w.length,bX=bW.w,bY=V.w;for(var i=0;i<N;i++)bX[i]=Math.log(1+Math.exp(bY[i]));this.out_act=bW;return this.out_act},backward:function(){var V=this.in_act,bZ=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++){var x=V.w[i];V.dw[i]=bZ.dw[i]*(1/(1+Math.exp(-x)))}},getParamsAndGrads:function(){return[]},toJSON:function(){var cA={};cA.out_depth=this.out_depth;cA.out_sx=this.out_sx;cA.out_sy=this.out_sy;cA.layer_type=this.layer_type;return cA},fromJSON:function(cB){this.out_depth=cB.out_depth;this.out_sx=cB.out_sx;this.out_sy=cB.out_sy;this.layer_type=cB.layer_type}};function l(x){var y=Math.exp(2*x);return (y-1)/(y+1)}m.prototype={forward:function(V){this.in_act=V;var cD=V.cloneAndZero(),N=V.w.length;for(var i=0;i<N;i++)cD.w[i]=l(V.w[i]);this.out_act=cD;return this.out_act},backward:function(){var V=this.in_act,cE=this.out_act,N=V.w.length;V.dw=_.zeros(N);for(var i=0;i<N;i++){var cF=cE.w[i];V.dw[i]=(1.0-cF*cF)*cE.dw[i]}},getParamsAndGrads:function(){return[]},toJSON:function(){var cG={};cG.out_depth=this.out_depth;cG.out_sx=this.out_sx;cG.out_sy=this.out_sy;cG.layer_type=this.layer_type;return cG},fromJSON:function(cH){this.out_depth=cH.out_depth;this.out_sx=cH.out_sx;this.out_sy=cH.out_sy;this.layer_type=cH.layer_type}};_.SoftplusLayer=k;_.MishLayer=B;_.PiLULayer=h;_.DoubleReLULayer=g;_.TanhLayer=m;_.LeakyReluLayer=b;_.EluLayer=c;_.FReluLayer=d;_.SwishLayer=e;_.PLULayer=f})(convnetjs);(function(cI){"use strict";var cJ=cI.assert,_c=function(){this.layers=[]};_c.prototype={makeLayers:function(cK){cJ(cK.length>=2,'Error! At least one input layer and one loss layer are required.');cJ(cK[0].type=='input','Error! First layer must be the input layer, to declare size of inputs');cK=function(){var cM=[];for(var i=0;i<cK.length;i++){var cN=cK[i];cN.type=='softmax'||cN.type=='svm'&&cM.push({type:'fc',num_neurons:cN.num_classes});cN.type=='regression'&&cM.push({type:'fc',num_neurons:cN.num_neurons});(cN.type=='fc'||cN.type=='conv')&&A(cN.bias_pref)&&(cN.bias_pref=0.0,!A(cN.activation)&&(cN.activation=='relu'||cN.activation=='leaky_relu'||cN.activation=='elu'||cN.activation=='frelu'||cN.activation=='double_relu'||cN.activation=='pilu')&&(cN.bias_pref=0.1));cM.push(cN);if(!A(cN.activation))if(cN.activation=='relu')cM.push({type:'relu'});else if(cN.activation=='leaky_relu')cM.push({type:'leaky_relu'});else if(cN.activation=='elu')cM.push({type:'elu'});else if(cN.activation=='frelu')cM.push({type:'frelu'});else if(cN.activation=='swish')cM.push({type:'swish'});else if(cN.activation=='plu')cM.push({type:'plu'});else if(cN.activation=='double_relu')cM.push({type:'double_relu'});else if(cN.activation=='softplus')cM.push({type:'softplus'});else if(cN.activation=='pilu')cM.push({type:'pilu'});else if(cN.activation=='sigmoid')cM.push({type:'sigmoid'});else if(cN.activation=='tanh')cM.push({type:'tanh'});else if(cN.activation=='maxout'){var cO=cN.group_size!=='undefined'?cN.group_size:2;cM.push({type:'maxout',group_size:cO})}else console.log('ERROR unsupported activation '+cN.activation);!A(cN.drop_prob)&&cN.type!=='dropout'&&cM.push({type:'dropout',drop_prob:cN.drop_prob})}return cM}(cK);this.layers=[];for(var i=0;i<cK.length;i++){var _C=cK[i];if(i>0){var cL=this.layers[i-1];_C.in_sx=cL.out_sx;_C.in_sy=cL.out_sy;_C.in_depth=cL.out_depth}switch(_C.type) {case 'fc':this.layers.push(new cI.FullyConnLayer(_C));break;case 'lrn':this.layers.push(new cI.LocalResponseNormalizationLayer(_C));break;case 'dropout':this.layers.push(new cI.DropoutLayer(_C));break;case 'input':this.layers.push(new cI.InputLayer(_C));break;case 'softmax':this.layers.push(new cI.SoftmaxLayer(_C));break;case 'regression':this.layers.push(new cI.RegressionLayer(_C));break;case 'conv':this.layers.push(new cI.ConvLayer(_C));break;case 'pool':this.layers.push(new cI.PoolLayer(_C));break;case 'relu':this.layers.push(new cI.ReluLayer(_C));break;case 'leaky_relu':this.layers.push(new cI.LeakyReluLayer(_C));break;case 'elu':this.layers.push(new cI.EluLayer(_C));break;case 'frelu':this.layers.push(new cI.FReluLayer(_C));break;case 'swish':this.layers.push(new cI.SwishLayer(_C));break;case 'plu':this.layers.push(new cI.PLULayer(_C));break;case 'double_relu':this.layers.push(new cI.DoubleRELULayer(_C));break;case 'softplus':this.layers.push(new cI.SoftplusLayer(_C));break;case 'pilu':this.layers.push(new cI.PiLULayer(_C));break;case 'sigmoid':this.layers.push(new cI.SigmoidLayer(_C));break;case 'tanh':this.layers.push(new cI.TanhLayer(_C));break;case 'maxout':this.layers.push(new cI.MaxoutLayer(_C));break;case 'svm':this.layers.push(new cI.SVMLayer(_C));break;default:console.log('ERROR: UNRECOGNIZED LAYER TYPE: '+_C.type)}}},forward:function(V,cP){A((cP))&&(cP=!1);var cQ=this.layers[0].forward(V,cP);for(var i=1;i<this.layers.length;i++)cQ=this.layers[i].forward(cQ,cP);return cQ},getCostLoss:function(V,y){this.forward(V,!1);var N=this.layers.length;return this.layers[N-1].backward(y)},backward:function(y){var N=this.layers.length,cR=this.layers[N-1].backward(y);for(var i=N-2;i>=0;i--)this.layers[i].backward();return cR},getParamsAndGrads:function(){var cS=[];for(var i=0;i<this.layers.length;i++){var cT=this.layers[i].getParamsAndGrads();for(var j=0;j<cT.length;j++)cS.push(cT[j])}return cS},getPrediction:function(){var S=this.layers[this.layers.length-1],p=S.out_act.w,cU=p[0],cV=0;cJ(S.layer_type=='softmax','getPrediction function assumes softmax as last layer of the net!');for(var i=1;i<p.length;i++)p[i]>cU&&(cU=p[i],cV=i);return cV},toJSON:function(){var cW={};cW.layers=[];for(var i=0;i<this.layers.length;i++)cW.layers.push(this.layers[i].toJSON());return cW},fromJSON:function(cX){this.layers=[];for(var i=0;i<cX.layers.length;i++){var cY=cX.layers[i],t=cY.layer_type,L;t=='input'&&(L=new cI.InputLayer());t=='relu'&&(L=new cI.ReluLayer());t=='leaky-relu'&&(L=new cI.LeakyReluLayer());t=='elu'&&(L=new cI.EluLayer());t=='frelu'&&(L=new cI.FReluLayer());t=='swish'&&(L=new cI.SwishLayer());t=='plu'&&(L=new cI.PLULayer());t=='double_relu'&&(L=new cI.DoubleRELULayer());t=='softplus'&&(L=new cI.SoftplusLayer());t=='pilu'&&(L=new cI.PiLULayer());t=='sigmoid'&&(L=new cI.SigmoidLayer());t=='tanh'&&(L=new cI.TanhLayer());t=='dropout'&&(L=new cI.DropoutLayer());t=='conv'&&(L=new cI.ConvLayer());t=='pool'&&(L=new cI.PoolLayer());t=='lrn'&&(L=new cI.LocalResponseNormalizationLayer());t=='softmax'&&(L=new cI.SoftmaxLayer());t=='regression'&&(L=new cI.RegressionLayer());t=='fc'&&(L=new cI.FullyConnLayer());t=='maxout'&&(L=new cI.MaxoutLayer());t=='svm'&&(L=new cI.SVMLayer());L.fromJSON(cY);this.layers.push(L)}}};cI.Net=_c})(convnetjs);