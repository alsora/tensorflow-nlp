/**
* TensorFlow Text node.js SDK
* @author Alberto Soragna (alberto dot soragna at gmail dot com)
*/
(function() {
  var TensorflowJS;
  TensorflowJS = {



    initialize: function(config) {
      for (var attrname in config) { this.config[attrname] = config[attrname]; }
      return this.config;
    }

  };

  module.exports = TensorflowJS;

}).call(this);