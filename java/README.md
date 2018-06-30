## Tensorflow-NLP in Java 

The Java library allows to perform inference on already trained models.

### Usage:

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








