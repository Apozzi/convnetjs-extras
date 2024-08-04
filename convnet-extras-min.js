!function(t){"use strict";t.Vol;var s=function(t){t=t||{};this.alpha=t.alpha?t.alpha:.01,this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="leaky_relu"};s.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=0;o<a;o++)r[o]<0&&(r[o]=r[o]*this.alpha);return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++)e.w[r]<=0?s.dw[r]=e.dw[r]*this.alpha:s.dw[r]=e.dw[r]},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var e=function(t){t=t||{};this.alpha=t.alpha?t.alpha:.01,this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="elu"};e.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=0;o<a;o++)r[o]<=0&&(r[o]=this.alpha*(Math.exp(r[o])-1));return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++)e.w[r]<=0?s.dw[r]=e.dw[r]*Math.exp(e.dw[r])*this.alpha:s.dw[r]=e.dw[r]},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var a=function(t){t=t||{};this.b=t.b?t.b:.05,this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="frelu"};a.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=0;o<a;o++)r[o]<=0?r[o]=this.b:r[o]=r[o]+this.b;return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++)e.w[r]<=0?s.dw[r]=0:s.dw[r]=e.dw[r]},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var r=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="swish"};r.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=t.w,i=0;i<a;i++)r[i]=o[i]/(1+Math.exp(-o[i]));return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=e.w[r];s.dw[r]=(s.w[r]*(o-o*o)+o)*e.dw[r]}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var o=function(t){t=t||{};this.alpha=t.alpha?t.alpha:.01,this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="plu"};o.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=(t.w,0);o<a;o++)r[o]>1?r[o]=(r[o]-1)*this.alpha+1:r[o]<-1&&(r[o]=(r[o]+1)*this.alpha-1);return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++)e.w[r]<-1||e.w[r]>1?s.dw[r]=e.dw[r]*this.alpha:s.dw[r]=e.dw[r]},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var i=function(t){t=t||{};this.alpha=t.alpha?t.alpha:.5,this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="double_relu"};i.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=(t.w,0);o<a;o++)r[o]<this.alpha?r[o]=r[o]+this.alpha:r[o]>this.alpha&&(r[o]=r[o]-this.alpha);return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++)Math.abs(e.w[r])>this.alpha?s.dw[r]=e.dw[r]*this.alpha:s.dw[r]=e.dw[r]},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var h=function(t){t=t||{};this.alpha=t.alpha?t.alpha:1.5,this.beta=t.beta?t.beta:3,this.gamma=t.gamma?t.gamma:1,this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="pilu"};h.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=t.w,i=0;i<a;i++)o[i]>gamma?r[i]=alpha*o[i]+gamma*(1-alpha):r[i]=beta*o[i]+gamma*(1-beta);return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++)s.w[r]>gamma?s.dw[r]=alpha*e.dw[r]:s.dw[r]=beta*e.dw[r]},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var u=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="mish"};u.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=t.w,i=0;i<a;i++)r[i]=o[i]*Math.tanh(Math.log(1+Math.exp(o[i])));return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r],i=Math.log(1+Math.exp(o)),h=Math.tanh(i),u=1-Math.pow(h,2),n=1/(1+Math.exp(-o));s.dw[r]=e.dw[r]*(h+o*u*n)}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var n=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.lower=t.lower||.01,this.upper=t.upper||.1,this.layer_type="rrelu",this.alpha=null};n.prototype={forward:function(t,s){this.in_act=t;var e=t.clone(),a=t.w.length;this.alpha=s?Math.random()*(this.upper-this.lower)+this.lower:(this.upper+this.lower)/2;for(var r=0;r<a;r++){var o=t.w[r];e.w[r]=o>0?o:this.alpha*o}return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r];s.dw[r]=e.dw[r]*(o>0?1:this.alpha)}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t.lower=this.lower,t.upper=this.upper,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type,this.lower=t.lower,this.upper=t.upper}};var _=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="gish"};_.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=t.w,i=0;i<a;i++)r[i]=o[i]*Math.log(2-Math.exp(-Math.exp(o[i])));return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r],i=2-Math.exp(-Math.exp(o)),h=o*(Math.exp(-Math.exp(o)+o)/i),u=Math.log(i);s.dw[r]=e.dw[r]*(h+u)}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var y=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="logish"};y.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=t.w,i=0;i<a;i++){var h=1/(1+Math.exp(-o[i]));r[i]=o[i]*Math.tanh(h)}return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=(this.out_act,s.w.length);s.dw=t.zeros(e);for(var a=0;a<e;a++);},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var p=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="softplus"};function l(t){const s=t.map((t=>Math.exp(-t))),e=s.reduce(((t,s)=>t+s),0);return s.map((t=>t/e))}function c(t,s,e){return t[s]*((s===e?1:0)-t[e])}p.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=e.w,o=t.w,i=0;i<a;i++)r[i]=Math.log(1+Math.exp(o[i]));return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r];s.dw[r]=e.dw[r]*(1/(1+Math.exp(-o)))}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var d=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="softmin"};d.prototype={forward:function(t,s){this.in_act=t;var e=t.clone();e.w;return l(t.w),this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=l(s.w),o=0;o<a;o++)for(var i=0;i<a;i++)s.dw[o]+=e.dw[i]*c(r,i,o)},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var f=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="softsign"};f.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=0;r<a;r++)e.w[r]=t.w[r]/(1+Math.abs(t.w[r]));return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r],i=1/Math.pow(1+Math.abs(o),2);s.dw[r]=e.dw[r]*i}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var w=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.lambda=t.lambda,this.layer_type="softshrink"};w.prototype={forward:function(t,s){this.in_act=t;for(var e,a,r=t.clone(),o=t.w.length,i=(r.w,t.w,0);i<o;i++)r.w[i]=(e=t.w[i],a=this.lambda,e>a?e-a:e<-a?e+a:0);return this.out_act=r,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r];s.dw[r]=e.dw[r]*(o>this.lambda||o<-this.lambda?1:0)}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var v=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="gelu"},x=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.alpha=t.alpha||.01,this.layer_type="prelu"};x.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=0;r<a;r++){var o=t.w[r];e.w[r]=o>0?o:this.alpha*o}return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r];s.dw[r]=e.dw[r]*(o>0?1:this.alpha)}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.alpha=this.alpha,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.alpha=t.alpha,this.layer_type=t.layer_type}},v.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=0;r<a;r++){var o=t.w[r];e.w[r]=.5*o*(1+Math.erf(o/Math.sqrt(2)))}return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);const r=t=>Math.exp(-.5*t*t)/Math.sqrt(2*Math.PI),o=t=>.5*(1+Math.erf(t/Math.sqrt(2)));for(var i=0;i<a;i++){var h=s.w[i];s.dw[i]=e.dw[i]*(o(h)+h*r(h))}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var g=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.lambda=t.lambda,this.layer_type="hardshrink"};g.prototype={forward:function(t,s){this.in_act=t;for(var e=t.clone(),a=t.w.length,r=0;r<a;r++){var o=t.w[r];e.w[r]=o*(o>this.lambda||o<-this.lambda?1:0)}return this.out_act=e,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=s.w[r];s.dw[r]=e.dw[r]*(o>this.lambda||o<-this.lambda?1:0)}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}};var m=function(t){t=t||{};this.out_sx=t.in_sx,this.out_sy=t.in_sy,this.out_depth=t.in_depth,this.layer_type="tanh"};m.prototype={forward:function(t,s){this.in_act=t;for(var e,a,r=t.cloneAndZero(),o=t.w.length,i=0;i<o;i++)r.w[i]=(e=t.w[i],a=void 0,((a=Math.exp(2*e))-1)/(a+1));return this.out_act=r,this.out_act},backward:function(){var s=this.in_act,e=this.out_act,a=s.w.length;s.dw=t.zeros(a);for(var r=0;r<a;r++){var o=e.w[r];s.dw[r]=(1-o*o)*e.dw[r]}},getParamsAndGrads:function(){return[]},toJSON:function(){var t={};return t.out_depth=this.out_depth,t.out_sx=this.out_sx,t.out_sy=this.out_sy,t.layer_type=this.layer_type,t},fromJSON:function(t){this.out_depth=t.out_depth,this.out_sx=t.out_sx,this.out_sy=t.out_sy,this.layer_type=t.layer_type}},t.SoftplusLayer=p,t.SoftminLayer=d,t.SoftsignLayer=f,t.SoftshrinkLayer=w,t.HardshrinkLayer=g,t.MishLayer=u,t.GishLayer=_,t.PiLULayer=h,t.DoubleReLULayer=i,t.GeluLayer=v,t.PReluLayer=x,t.RReluLayer=n,t.TanhLayer=m,t.LeakyReluLayer=s,t.EluLayer=e,t.FReluLayer=a,t.SwishLayer=r,t.LogishLayer=y,t.PLULayer=o}(convnetjs),function(t){"use strict";t.Vol;var s=t.assert,e=function(t){this.layers=[]};e.prototype={makeLayers:function(e){s(e.length>=2,"Error! At least one input layer and one loss layer are required."),s("input"===e[0].type,"Error! First layer must be the input layer, to declare size of inputs");e=function(){for(var t=[],s=0;s<e.length;s++){var a=e[s];if("hardshrink"!==a.type&&"softmin"!==a.type&&"softsign"!==a.type&&"softshrink"!==a.type&&"softplus"!==a.type&&"softmax"!==a.type&&"svm"!==a.type||t.push({type:"fc",num_neurons:a.num_classes}),"regression"===a.type&&t.push({type:"fc",num_neurons:a.num_neurons}),"fc"!==a.type&&"conv"!==a.type||void 0!==a.bias_pref||(a.bias_pref=0,void 0===a.activation||"rrelu"!==a.activation&&"prelu"!==a.activation&&"gelu"!==a.activation&&"relu"!==a.activation&&"leaky_relu"!==a.activation&&"elu"!==a.activation&&"frelu"!==a.activation&&"double_relu"!==a.activation&&"pilu"!==a.activation||(a.bias_pref=.1)),t.push(a),void 0!==a.activation)if("relu"===a.activation)t.push({type:"relu"});else if("leaky_relu"===a.activation)t.push({type:"leaky_relu"});else if("elu"===a.activation)t.push({type:"elu"});else if("frelu"===a.activation)t.push({type:"frelu"});else if("gelu"===a.activation)t.push({type:"gelu"});else if("prelu"===a.activation)t.push({type:"prelu"});else if("rrelu"===a.activation)t.push({type:"rrelu"});else if("swish"===a.activation)t.push({type:"swish"});else if("gish"===a.activation)t.push({type:"gish"});else if("logish"===a.activation)t.push({type:"logish"});else if("plu"===a.activation)t.push({type:"plu"});else if("double_relu"===a.activation)t.push({type:"double_relu"});else if("softplus"===a.activation)t.push({type:"softplus"});else if("softmin"===a.activation)t.push({type:"softmin"});else if("softsign"===a.activation)t.push({type:"softsign"});else if("softshrink"===a.activation)t.push({type:"softshrink"});else if("hardshrink"===a.activation)t.push({type:"hardshrink"});else if("pilu"===a.activation)t.push({type:"pilu"});else if("sigmoid"===a.activation)t.push({type:"sigmoid"});else if("tanh"===a.activation)t.push({type:"tanh"});else if("maxout"===a.activation){var r="undefined"!==a.group_size?a.group_size:2;t.push({type:"maxout",group_size:r})}else console.log("ERROR unsupported activation "+a.activation);void 0!==a.drop_prob&&"dropout"!==a.type&&t.push({type:"dropout",drop_prob:a.drop_prob})}return t}(),this.layers=[];for(var a=0;a<e.length;a++){var r=e[a];if(a>0){var o=this.layers[a-1];r.in_sx=o.out_sx,r.in_sy=o.out_sy,r.in_depth=o.out_depth}switch(r.type){case"fc":this.layers.push(new t.FullyConnLayer(r));break;case"lrn":this.layers.push(new t.LocalResponseNormalizationLayer(r));break;case"dropout":this.layers.push(new t.DropoutLayer(r));break;case"input":this.layers.push(new t.InputLayer(r));break;case"softmax":this.layers.push(new t.SoftmaxLayer(r));break;case"logsoftmax":this.layers.push(new t.LogSoftmaxLayer(r));break;case"softmin":this.layers.push(new t.SoftminLayer(r));break;case"softsign":this.layers.push(new t.SoftsignLayer(r));break;case"softshrink":this.layers.push(new t.SoftshrinkLayer(r));break;case"hardshrink":this.layers.push(new t.HardshrinkLayer(r));break;case"regression":this.layers.push(new t.RegressionLayer(r));break;case"conv":this.layers.push(new t.ConvLayer(r));break;case"pool":this.layers.push(new t.PoolLayer(r));break;case"relu":this.layers.push(new t.ReluLayer(r));break;case"gelu":this.layers.push(new t.GeluLayer(r));break;case"rrelu":this.layers.push(new t.RReluLayer(r));break;case"prelu":this.layers.push(new t.PReluLayerr(r));break;case"leaky_relu":this.layers.push(new t.LeakyReluLayer(r));break;case"elu":this.layers.push(new t.EluLayer(r));break;case"frelu":this.layers.push(new t.FReluLayer(r));break;case"swish":this.layers.push(new t.SwishLayer(r));break;case"gish":this.layers.push(new t.GishLayer(r));break;case"logish":this.layers.push(new t.LogishLayer(r));break;case"plu":this.layers.push(new t.PLULayer(r));break;case"double_relu":this.layers.push(new t.DoubleRELULayer(r));break;case"softplus":this.layers.push(new t.SoftplusLayer(r));break;case"pilu":this.layers.push(new t.PiLULayer(r));break;case"sigmoid":this.layers.push(new t.SigmoidLayer(r));break;case"tanh":this.layers.push(new t.TanhLayer(r));break;case"maxout":this.layers.push(new t.MaxoutLayer(r));break;case"svm":this.layers.push(new t.SVMLayer(r));break;default:console.log("ERROR: UNRECOGNIZED LAYER TYPE: "+r.type)}}},forward:function(t,s){void 0===s&&(s=!1);for(var e=this.layers[0].forward(t,s),a=1;a<this.layers.length;a++)e=this.layers[a].forward(e,s);return e},getCostLoss:function(t,s){this.forward(t,!1);var e=this.layers.length;return this.layers[e-1].backward(s)},backward:function(t){for(var s=this.layers.length,e=this.layers[s-1].backward(t),a=s-2;a>=0;a--)this.layers[a].backward();return e},getParamsAndGrads:function(){for(var t=[],s=0;s<this.layers.length;s++)for(var e=this.layers[s].getParamsAndGrads(),a=0;a<e.length;a++)t.push(e[a]);return t},getPrediction:function(){var t=this.layers[this.layers.length-1];s("softmax"===t.layer_type,"getPrediction function assumes softmax as last layer of the net!");for(var e=t.out_act.w,a=e[0],r=0,o=1;o<e.length;o++)e[o]>a&&(a=e[o],r=o);return r},toJSON:function(){for(var t={layers:[]},s=0;s<this.layers.length;s++)t.layers.push(this.layers[s].toJSON());return t},fromJSON:function(s){this.layers=[];for(var e=0;e<s.layers.length;e++){var a,r=s.layers[e],o=r.layer_type;"input"===o&&(a=new t.InputLayer),"relu"===o&&(a=new t.ReluLayer),"leaky-relu"===o&&(a=new t.LeakyReluLayer),"elu"===o&&(a=new t.EluLayer),"frelu"===o&&(a=new t.FReluLayer),"gelu"===o&&(a=new t.GeluLayer),"frelu"===o&&(a=new t.FReluLayer),"rrelu"===o&&(a=new t.RReluLayer),"swish"===o&&(a=new t.SwishLayer),"gish"===o&&(a=new t.GishLayer),"logish"===o&&(a=new t.LogishLayer),"plu"===o&&(a=new t.PLULayer),"double_relu"===o&&(a=new t.DoubleRELULayer),"softplus"===o&&(a=new t.SoftplusLayer),"softmin"===o&&(a=new t.SoftminLayer),"softsign"===o&&(a=new t.SoftsignLayer),"softshrink"===o&&(a=new t.SoftshrinkLayer),"hardshrink"===o&&(a=new t.HardshrinkLayer),"pilu"===o&&(a=new t.PiLULayer),"sigmoid"===o&&(a=new t.SigmoidLayer),"tanh"===o&&(a=new t.TanhLayer),"dropout"===o&&(a=new t.DropoutLayer),"conv"===o&&(a=new t.ConvLayer),"pool"===o&&(a=new t.PoolLayer),"lrn"===o&&(a=new t.LocalResponseNormalizationLayer),"softmax"===o&&(a=new t.SoftmaxLayer),"logsoftmax"===o&&(a=new t.LogSoftmaxLayer),"regression"===o&&(a=new t.RegressionLayer),"fc"===o&&(a=new t.FullyConnLayer),"maxout"===o&&(a=new t.MaxoutLayer),"svm"===o&&(a=new t.SVMLayer),a.fromJSON(r),this.layers.push(a)}}},t.Net=e}(convnetjs);