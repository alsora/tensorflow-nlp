package serving_utils;

import java.io.FileNotFoundException;
import java.time.Instant;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import data_utils.TextProcessor;


public class SequenceTaggingModel {
	
	private TextProcessor textProcessor;
	private SavedModelBundle model;
	private Session session;
	private Graph graph;
	private Tensor dropout_keep_prob;
	private int sequenceLength;
	
	public SequenceTaggingModel(String modelDirPath) throws FileNotFoundException {
		this(modelDirPath, 9);
	}
	
	public SequenceTaggingModel(String modelDirPath, int sequenceLength) throws FileNotFoundException{
		
		if (modelDirPath.length() == 0) {
			throw new FileNotFoundException("Incorrect modelDir format");
		}
		
		if (modelDirPath.substring(modelDirPath.length() - 1) != "/") {
			modelDirPath += "/";
		}
		
		String vocabWordsFile = modelDirPath + "vocab_words";
		String vocabLabelsFile = modelDirPath + "vocab_labels";
		this.sequenceLength = sequenceLength;
		this.textProcessor = new TextProcessor(this.sequenceLength, vocabWordsFile, vocabLabelsFile);
		
		
		String savedModelFolder = modelDirPath + "saved";
		this.model = SavedModelBundle.load(savedModelFolder, "serve");
		this.session = model.session();
		this.graph = model.graph();
		
		Float prob = (float) 1.0;
		dropout_keep_prob = Tensor.create(prob);
			
	}
	
	public String[] inference(String sentence) {
		
		int batch_size = 1;

		String[][] data =  new String[batch_size][this.sequenceLength];; 
		
		data[0] = textProcessor.splitPadSentence(sentence);
		
		
		Tensor input_t = textProcessor.textToInputTensor(data);

		
		List<Tensor<?>> output = session
									.runner()
									.feed("input_x", input_t)
									.feed("dropout_keep_prob", dropout_keep_prob)
									.fetch("output/predictions")
									.run();
		
		
		Tensor prediction = output.get(0);
				
		int[][] copy_prediction = new int[batch_size][this.sequenceLength];
		prediction.copyTo(copy_prediction);
		
		String[] string_pred = new String[this.sequenceLength];
		
		for (int i = 0; i < this.sequenceLength; i++) {
			int int_pred = copy_prediction[0][i];
						
			string_pred[i] = textProcessor.labelToString(int_pred);
			
		}
		
		
		return string_pred;
	}
	
	

}
