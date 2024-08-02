(function(global) {
  "use strict";
  var Vol = global.Vol; 
  

  var LeakyReluLayer = function(opt) {
      var opt = opt || {};
  
      this.alpha = opt.alpha ? opt.alpha : 0.01;
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'leaky_relu';
    }
    LeakyReluLayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        for(var i=0;i<N;i++) { 
          if(V2w[i] < 0) V2w[i] = V2w[i]*this.alpha; 
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          if(V2.w[i] <= 0) V.dw[i] = V2.dw[i]*this.alpha; 
          else V.dw[i] = V2.dw[i];
        }
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }

    // a(e^x-1) if x<=0
    // x if x > 0
    var EluLayer = function(opt) {
      var opt = opt || {};
  
      this.alpha = opt.alpha ? opt.alpha : 0.01;
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'elu';
    }
    EluLayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        for(var i=0;i<N;i++) { 
          if(V2w[i] <= 0) V2w[i] = this.alpha*(Math.exp(V2w[i])-1);
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          if(V2.w[i] <= 0) V.dw[i] = V2.dw[i]*Math.exp(V2.dw[i])*this.alpha; 
          else V.dw[i] = V2.dw[i];
        }
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }

    // from: https://arxiv.org/pdf/1706.08098
    // b if x<=0
    // x + b if x > 0
    var FReluLayer = function(opt) {
      var opt = opt || {};
  
      this.b = opt.b ? opt.b : 0.05;
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'frelu';
    }
    FReluLayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        for(var i=0;i<N;i++) { 
          if(V2w[i] <= 0) V2w[i] = this.b;
          else V2w[i] = V2w[i] + this.b;
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          if(V2.w[i] <= 0) V.dw[i] = 0; 
          else V.dw[i] = V2.dw[i];
        }
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }


    var SwishLayer = function(opt) {
      var opt = opt || {};
  
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'swish';
    }
    SwishLayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;
        for(var i=0;i<N;i++) {
          V2w[i] = Vw[i]/(1.0+Math.exp(-Vw[i]));
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          var v2wi = V2.w[i];
          V.dw[i] = (V.w[i]*(v2wi - v2wi*v2wi) + v2wi) * V2.dw[i];
        }
        
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }

    // https://ar5iv.labs.arxiv.org/html/1809.09534
    // 𝑃𝐿𝑈(𝑥) ≡ 𝑚𝑎𝑥(𝛼(𝑥+𝑐)−𝑐,𝑚𝑖𝑛(𝛼(𝑥−𝑐)+𝑐,𝑥))
    var PLULayer = function(opt) {
      var opt = opt || {};
  
      this.alpha= opt.alpha ? opt.alpha : 0.01;
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'plu';
    }
    PLULayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;
        for(var i=0;i<N;i++) { 
           if (V2w[i] > 1) V2w[i] = (V2w[i]-1)*this.alpha+1;
           else if(V2w[i] < -1) V2w[i] = (V2w[i]+1)*this.alpha-1;
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          if(V2.w[i] < -1 || V2.w[i] > 1) V.dw[i] = V2.dw[i]*this.alpha; 
          else V.dw[i] = V2.dw[i];
        }
        
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }

    // https://arxiv.org/pdf/2108.00700
    var DoubleReLULayer = function(opt) {
      var opt = opt || {};
  
      this.alpha = opt.alpha ? opt.alpha : 0.5;
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'double_relu';
    }
    DoubleReLULayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;
        for(var i=0;i<N;i++) { 
           if (V2w[i] <  this.alpha) V2w[i] = V2w[i] + this.alpha;
           else if(V2w[i] > this.alpha) V2w[i] = V2w[i] - this.alpha;
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          if(Math.abs(V2.w[i]) > this.alpha) V.dw[i] = V2.dw[i]*this.alpha; 
          else V.dw[i] = V2.dw[i];
        }
        
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }


    // https://arxiv.org/pdf/2108.00700
    var PiLULayer = function(opt) {
      var opt = opt || {};
  
      this.alpha = opt.alpha ? opt.alpha : 1.5;
      this.beta = opt.beta ? opt.beta : 3;
      this.gamma = opt.gamma ? opt.gamma : 1;
      this.out_sx = opt.in_sx;
      this.out_sy = opt.in_sy;
      this.out_depth = opt.in_depth;
      this.layer_type = 'pilu';
    }
    PiLULayer.prototype = {
      forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;
        for(var i=0;i<N;i++) { 
          if (Vw[i] > gamma) {
            V2w[i] = alpha * Vw[i] + gamma * (1 - alpha);
          } else {
            V2w[i] = beta * Vw[i] + gamma * (1 - beta);
          }
        }
        this.out_act = V2;
        return this.out_act;
      },
      backward: function() {
        var V = this.in_act; 
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N); 
        for(var i=0;i<N;i++) {
          if (V.w[i] > gamma) {
            V.dw[i] = alpha * V2.dw[i];
          } else {
            V.dw[i] = beta * V2.dw[i];
          }
        }
        
      },
      getParamsAndGrads: function() {
        return [];
      },
      toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
      },
      fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type; 
      }
    }


  //https://arxiv.org/pdf/1908.08681
  var MishLayer = function(opt) {
    var opt = opt || {};

    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'mish';
  }
  
  MishLayer.prototype = {
      forward: function(V, is_training) {
          this.in_act = V;
          var V2 = V.clone();
          var N = V.w.length;
          var V2w = V2.w;
          var Vw = V.w;
  
          for (var i = 0; i < N; i++) {
              V2w[i] = Vw[i] * Math.tanh(Math.log(1 + Math.exp(Vw[i])));
          }
  
          this.out_act = V2;
          return this.out_act;
      },
      backward: function() {
          var V = this.in_act;
          var V2 = this.out_act;
          var N = V.w.length;
          V.dw = global.zeros(N);
  
          for (var i = 0; i < N; i++) {
              var x = V.w[i];
              var sp = Math.log(1 + Math.exp(x));
              var tsp = Math.tanh(sp);
              var grad_tanh_sp = 1 - Math.pow(tsp, 2);  // sech^2(sp)
              var sigmoid_x = 1 / (1 + Math.exp(-x));
  
              V.dw[i] = V2.dw[i] * (tsp + x * grad_tanh_sp * sigmoid_x);
          }
      },
      getParamsAndGrads: function() {
          return [];
      },
      toJSON: function() {
          var json = {};
          json.out_depth = this.out_depth;
          json.out_sx = this.out_sx;
          json.out_sy = this.out_sy;
          json.layer_type = this.layer_type;
          return json;
      },
      fromJSON: function(json) {
          this.out_depth = json.out_depth;
          this.out_sx = json.out_sx;
          this.out_sy = json.out_sy;
          this.layer_type = json.layer_type;
      }
  }

  var GishLayer = function(opt) {
    var opt = opt || {};
  
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'gish';
  }
  GishLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;

        for (var i = 0; i < N; i++) {
            V2w[i] = Vw[i] * Math.log(2 - Math.exp(-Math.exp(Vw[i])));
        }

        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
        var V = this.in_act;
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N);
    
        for (var i = 0; i < N; i++) {
            var x = V.w[i];
            var exp_neg_exp_x = Math.exp(-Math.exp(x));
            var numerator = Math.exp(-Math.exp(x) + x);
            var denominator = 2 - exp_neg_exp_x;
    
            var term1 = x * (numerator / denominator);
            var term2 = Math.log(denominator);
    
            V.dw[i] = V2.dw[i] * (term1 + term2);
        }
    },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

  //TODO
var SmishLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'smish';
}

SmishLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;

        for (var i = 0; i < N; i++) {
            var sigmoid_x = 1 / (1 + Math.exp(-Vw[i]));
            V2w[i] = Vw[i] * Math.tanh(sigmoid_x);
        }

        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
        var V = this.in_act;
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N);

        for (var i = 0; i < N; i++) {
            //TODO
        }
    },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

var LogishLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'logish';
}
LogishLayer.prototype = {
  forward: function(V, is_training) {
      this.in_act = V;
      var V2 = V.clone();
      var N = V.w.length;
      var V2w = V2.w;
      var Vw = V.w;

      for (var i = 0; i < N; i++) {
          var sigmoid_x = 1 / (1 + Math.exp(-Vw[i]));
          V2w[i] = Vw[i] * Math.tanh(sigmoid_x);
      }

      this.out_act = V2;
      return this.out_act;
  },
  backward: function() {
      var V = this.in_act;
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N);

      for (var i = 0; i < N; i++) {
          //TODO
      }
  },
  getParamsAndGrads: function() {
      return [];
  },
  toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      return json;
  },
  fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
  }
}

var SoftplusLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'softplus';
}
SoftplusLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;

        for (var i = 0; i < N; i++) {
            V2w[i] = Math.log(1 + Math.exp(Vw[i]));
        }
        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
        var V = this.in_act;
        var V2 = this.out_act;
        var N = V.w.length;
        V.dw = global.zeros(N);
        for (var i = 0; i < N; i++) {
            var x = V.w[i];
            V.dw[i] = V2.dw[i] * (1 / (1 + Math.exp(-x))); // sigmoid 
        }
    },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

function softmin(arr) {
  const expNegArr = arr.map(x => Math.exp(-x));
  const sumExpNegArr = expNegArr.reduce((a, b) => a + b, 0);
  return expNegArr.map(x => x / sumExpNegArr);
}

function softminDerivative(softminVec, i, j) {
  return softminVec[i] * ((i === j ? 1 : 0) - softminVec[j]);
}

var SoftminLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'softmin';
}
SoftminLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var V2w = V2.w;
        var Vw = V.w;

        V2w = softmin(Vw);
        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N);

      var softminVec = softmin(V.w);
      for (var i = 0; i < N; i++) {
          for (var j = 0; j < N; j++) {
              V.dw[i] += V2.dw[j] * softminDerivative(softminVec, j, i);
          }
      }
  },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}



var SoftsignLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'softsign';
}
SoftsignLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;

        for(var i=0;i<N;i++) { 
          V2.w[i] = V.w[i] / (1 + Math.abs(V.w[i]));
        }
        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N);
  
      for (var i = 0; i < N; i++) {
          var x = V.w[i];
          var softsign_derivative = 1 / Math.pow(1 + Math.abs(x), 2);
  
          V.dw[i] = V2.dw[i] * softsign_derivative;
      }
   },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

function softshrink(x, lambda) {
  if (x > lambda) {
      return x - lambda;
  }
  return x < -lambda ? x + lambda: 0;
}

var SoftshrinkLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.lambda = opt.lambda;
  this.layer_type = 'softshrink';
}
SoftshrinkLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;
        var V2w = V2.w;
        var Vw = V.w;

        for(var i=0;i<N;i++) { 
          V2.w[i] = softshrink(V.w[i], this.lambda);
        }
        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N);
  
      for (var i = 0; i < N; i++) {
          var x = V.w[i];
          V.dw[i] = V2.dw[i] * ((x > this.lambda || x < -this.lambda) ? 1 : 0);
      }
   },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

var HardshrinkLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.lambda = opt.lambda;
  this.layer_type = 'hardshrink';
}
HardshrinkLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;

        for(var i=0;i<N;i++) { 
          var x = V.w[i];
          V2.w[i] = x * ((x > this.lambda || x < -this.lambda) ? 1 : 0);
        }
        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N);
  
      for (var i = 0; i < N; i++) {
          var x = V.w[i];
          V.dw[i] = V2.dw[i] * ((x > this.lambda || x < -this.lambda) ? 1 : 0);
      }
   },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

function logSoftmax(x) {
  const maxVal = Math.max(...x);
  const expX = x.map(val => Math.exp(val - maxVal));
  const sumExpX = expX.reduce((a, b) => a + b, 0);
  return x.map(val => Math.log(expX[val - maxVal] / sumExpX));
}

var LogSoftmaxLayer = function(opt) {
  var opt = opt || {};

  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'logsoftmax';
}
LogSoftmaxLayer.prototype = {
    forward: function(V, is_training) {
        this.in_act = V;
        var V2 = V.clone();
        var N = V.w.length;

        V2.w = logSoftmax(V.w)
        this.out_act = V2;
        return this.out_act;
    },
    backward: function() {
      var V = this.in_act; 
      var V2 = this.out_act; 
      var N = V.w.length;
      V.dw = global.zeros(N);
  
      var expX = V.w.map(val => Math.exp(val));
      var sumExpX = expX.reduce((a, b) => a + b, 0);
      var softmaxValues = expX.map(val => val / sumExpX);
  
      for (var i = 0; i < N; i++) {
          var grad = softmaxValues[i] - (V2.dw[i] / sumExpX);
          V.dw[i] = grad;
      }
  },
    getParamsAndGrads: function() {
        return [];
    },
    toJSON: function() {
        var json = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    },
    fromJSON: function(json) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
    }
}

  function tanh(x) {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
  var TanhLayer = function(opt) {
    var opt = opt || {};

    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'tanh';
  }
  TanhLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var V2 = V.cloneAndZero();
      var N = V.w.length;
      for(var i=0;i<N;i++) { 
        V2.w[i] = tanh(V.w[i]);
      }
      this.out_act = V2;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act; 
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); 
      for(var i=0;i<N;i++) {
        var v2wi = V2.w[i];
        V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
      }
    },
    getParamsAndGrads: function() {
      return [];
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
    }
  }
  
  global.SoftplusLayer = SoftplusLayer;
  global.SoftminLayer = SoftminLayer;
  global.SoftsignLayer = SoftsignLayer;
  global.SoftshrinkLayer = SoftshrinkLayer;
  global.HardshrinkLayer = HardshrinkLayer;
  global.MishLayer = MishLayer;
  global.GishLayer = GishLayer;
  global.PiLULayer = PiLULayer;
  global.DoubleReLULayer = DoubleReLULayer;
  
  global.TanhLayer = TanhLayer;
  global.LeakyReluLayer = LeakyReluLayer;
  global.EluLayer = EluLayer;
  global.FReluLayer = FReluLayer;
  global.SwishLayer = SwishLayer;
  global.LogishLayer = LogishLayer;
  global.PLULayer = PLULayer;

})(convnetjs);

(function(global) {
  "use strict";
  var Vol = global.Vol; 
  var assert = global.assert;

  var Net = function(options) {
    this.layers = [];
  }

  Net.prototype = {
    

    makeLayers: function(defs) {

      assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
      assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

      var desugar = function() {
        var new_defs = [];
        for(var i=0;i<defs.length;i++) {
          var def = defs[i];
          
          if(def.type==='logsoftmax' || def.type==='softmax' || def.type==='svm') {
            new_defs.push({type:'fc', num_neurons: def.num_classes});
          }
          if(def.type==='regression') {
            new_defs.push({type:'fc', num_neurons: def.num_neurons});
          }

          if((def.type==='fc' || def.type==='conv') 
              && typeof(def.bias_pref) === 'undefined'){
            def.bias_pref = 0.0;
            if(typeof def.activation !== 'undefined' && (def.activation === 'relu' || def.activation === 'leaky_relu' 
            || def.activation === 'elu' || def.activation === 'frelu' || def.activation === 'double_relu'
            || def.activation === 'pilu')) {
              def.bias_pref = 0.1;
            }
          }

          new_defs.push(def);

          if(typeof def.activation !== 'undefined') {
            if(def.activation==='relu') { new_defs.push({type:'relu'}); }
            else if(def.activation==='leaky_relu') { new_defs.push({type:'leaky_relu'}); }
            else if(def.activation==='elu') { new_defs.push({type:'elu'}); }
            else if(def.activation==='frelu') { new_defs.push({type:'frelu'}); }
            else if(def.activation==='swish') { new_defs.push({type:'swish'}); }
            else if(def.activation==='gish') { new_defs.push({type:'gish'}); }
            else if(def.activation==='logish') { new_defs.push({type:'logish'}); }
            else if(def.activation==='plu') { new_defs.push({type:'plu'}); }
            else if(def.activation==='double_relu') { new_defs.push({type:'double_relu'}); }
            else if(def.activation==='softplus') { new_defs.push({type:'softplus'}); }
            else if(def.activation==='softmin') { new_defs.push({type:'softmin'}); }
            else if(def.activation==='softsign') { new_defs.push({type:'softsign'}); }
            else if(def.activation==='softshrink') { new_defs.push({type:'softshrink'}); }
            else if(def.activation==='hardshrink') { new_defs.push({type:'hardshrink'}); }
            else if(def.activation==='pilu') { new_defs.push({type:'pilu'}); }
            else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
            else if (def.activation==='tanh') { new_defs.push({type:'tanh'}); }
            else if (def.activation==='maxout') {
              var gs = def.group_size !== 'undefined' ? def.group_size : 2;
              new_defs.push({type:'maxout', group_size:gs});
            }
            else { console.log('ERROR unsupported activation ' + def.activation); }
          }
          if(typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') {
            new_defs.push({type:'dropout', drop_prob: def.drop_prob});
          }

        }
        return new_defs;
      }
      defs = desugar(defs);

      // create the layers
      this.layers = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        if(i>0) {
          var prev = this.layers[i-1];
          def.in_sx = prev.out_sx;
          def.in_sy = prev.out_sy;
          def.in_depth = prev.out_depth;
        }

        switch(def.type) {
          case 'fc': this.layers.push(new global.FullyConnLayer(def)); break;
          case 'lrn': this.layers.push(new global.LocalResponseNormalizationLayer(def)); break;
          case 'dropout': this.layers.push(new global.DropoutLayer(def)); break;
          case 'input': this.layers.push(new global.InputLayer(def)); break;
          case 'softmax': this.layers.push(new global.SoftmaxLayer(def)); break;
          case 'logsoftmax': this.layers.push(new global.LogSoftmaxLayer(def)); break;
          case 'softmin': this.layers.push(new global.SoftminLayer(def)); break;
          case 'softsign': this.layers.push(new global.SoftsignLayer(def)); break;
          case 'softshrink': this.layers.push(new global.SoftshrinkLayer(def)); break;
          case 'hardshrink': this.layers.push(new global.HardshrinkLayer(def)); break;
          case 'regression': this.layers.push(new global.RegressionLayer(def)); break;
          case 'conv': this.layers.push(new global.ConvLayer(def)); break;
          case 'pool': this.layers.push(new global.PoolLayer(def)); break;
          case 'relu': this.layers.push(new global.ReluLayer(def)); break;
          case 'leaky_relu': this.layers.push(new global.LeakyReluLayer(def)); break;
          case 'elu': this.layers.push(new global.EluLayer(def)); break;
          case 'frelu': this.layers.push(new global.FReluLayer(def)); break;
          case 'swish': this.layers.push(new global.SwishLayer(def)); break;
          case 'gish': this.layers.push(new global.GishLayer(def)); break;
          case 'logish': this.layers.push(new global.LogishLayer(def)); break;
          case 'plu': this.layers.push(new global.PLULayer(def)); break;
          case 'double_relu': this.layers.push(new global.DoubleRELULayer(def)); break;
          case 'softplus': this.layers.push(new global.SoftplusLayer(def)); break;
          case 'pilu': this.layers.push(new global.PiLULayer(def)); break;
          case 'sigmoid': this.layers.push(new global.SigmoidLayer(def)); break;
          case 'tanh': this.layers.push(new global.TanhLayer(def)); break;
          case 'maxout': this.layers.push(new global.MaxoutLayer(def)); break;
          case 'svm': this.layers.push(new global.SVMLayer(def)); break;
          default: console.log('ERROR: UNRECOGNIZED LAYER TYPE: ' + def.type);
        }
      }
    },

    forward: function(V, is_training) {
      if(typeof(is_training) === 'undefined') is_training = false;
      var act = this.layers[0].forward(V, is_training);
      for(var i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act, is_training);
      }
      return act;
    },

    getCostLoss: function(V, y) {
      this.forward(V, false);
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y);
      return loss;
    },
    
    backward: function(y) {
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y); 
      for(var i=N-2;i>=0;i--) { 
        this.layers[i].backward();
      }
      return loss;
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.layers.length;i++) {
        var layer_reponse = this.layers[i].getParamsAndGrads();
        for(var j=0;j<layer_reponse.length;j++) {
          response.push(layer_reponse[j]);
        }
      }
      return response;
    },
    getPrediction: function() {
      var S = this.layers[this.layers.length-1];
      assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

      var p = S.out_act.w;
      var maxv = p[0];
      var maxi = 0;
      for(var i=1;i<p.length;i++) {
        if(p[i] > maxv) { maxv = p[i]; maxi = i;}
      }
      return maxi; 
    },
    toJSON: function() {
      var json = {};
      json.layers = [];
      for(var i=0;i<this.layers.length;i++) {
        json.layers.push(this.layers[i].toJSON());
      }
      return json;
    },
    fromJSON: function(json) {
      this.layers = [];
      for(var i=0;i<json.layers.length;i++) {
        var Lj = json.layers[i]
        var t = Lj.layer_type;
        var L;
        if(t==='input') { L = new global.InputLayer(); }
        if(t==='relu') { L = new global.ReluLayer(); }
        if(t==='leaky-relu') { L = new global.LeakyReluLayer(); }
        if(t==='elu') { L = new global.EluLayer(); }
        if(t==='frelu') { L = new global.FReluLayer(); }
        if(t==='swish') { L = new global.SwishLayer(); }
        if(t==='gish') { L = new global.GishLayer(); }
        if(t==='logish') { L = new global.LogishLayer(); }
        if(t==='plu') { L = new global.PLULayer(); }
        if(t==='double_relu') { L = new global.DoubleRELULayer(); }
        if(t==='softplus') { L = new global.SoftplusLayer(); }
        if(t==='softmin') { L = new global.SoftminLayer(); }
        if(t==='softsign') { L = new global.SoftsignLayer(); }
        if(t==='softshrink') { L = new global.SoftshrinkLayer(); }
        if(t==='hardshrink') { L = new global.HardshrinkLayer(); }
        if(t==='pilu') { L = new global.PiLULayer(); }
        if(t==='sigmoid') { L = new global.SigmoidLayer(); }
        if(t==='tanh') { L = new global.TanhLayer(); }
        if(t==='dropout') { L = new global.DropoutLayer(); }
        if(t==='conv') { L = new global.ConvLayer(); }
        if(t==='pool') { L = new global.PoolLayer(); }
        if(t==='lrn') { L = new global.LocalResponseNormalizationLayer(); }
        if(t==='softmax') { L = new global.SoftmaxLayer(); }
        if(t==='logsoftmax') { L = new global.LogSoftmaxLayer(); }
        if(t==='regression') { L = new global.RegressionLayer(); }
        if(t==='fc') { L = new global.FullyConnLayer(); }
        if(t==='maxout') { L = new global.MaxoutLayer(); }
        if(t==='svm') { L = new global.SVMLayer(); }
        L.fromJSON(Lj);
        this.layers.push(L);
      }
    }
  }
  
  global.Net = Net;
})(convnetjs);

