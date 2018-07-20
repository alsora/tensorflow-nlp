# Tensorflow-NLP in Java 

The Java library allows to perform inference on already trained models.

## Installation

Follow these steps to install the Java bindings for Tensorflow and exploit the provided Java library.

  - Download the .jar file and the Java Native Interface (JNI).
  
        $ cd java
        $ bash script/download_tensorflow.sh

  - In your preferred IDE, add the .jar file to the Java project build path and link to it the native libraries contained in the jni folder.
  Then you can validate the installation by running the HelloTF.java example.


  - In alternative, if you want to work from command line, instead of using an IDE:

        $ javac -d bin -sourcepath src -cp lib/libtensorflow-1.8.0.jar src/main/HelloTF.java
        $ java -cp bin:lib/libtensorflow-1.8.0.jar -Djava.library.path=jni/$(uname -s | tr '[:upper:]' '[:lower:]') main.HelloTF


## Usage:

##### Trained model inference

Assuming to have trained a model using the provided Python library, it is possible to perform inference on it in Java with a source file like this.


    import serving_utils.SentenceClassificationModel;
    
    public class TFInference {
      public static void main(String[] args) throws Exception {
        SentenceClassificationModel model;
        String modelPath = "data/models/blstm_sentiment/"
        model =  new SentenceClassificationModel(modelPath);
        
        String text = "I'm Alberto and this is a beautiful day"
        
        String prediction = model.inference(text);
        
        System.out.println("Predicted class ---->" + prediction);
    }








