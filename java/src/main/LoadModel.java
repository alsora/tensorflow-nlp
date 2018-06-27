package main;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;



import data_utils.TextProcessor;


public class LoadModel {
	public static void main(String[] args) throws Exception {


		Tensor input_t;
		List<Tensor<?>> output;
		Tensor prediction;
		Tensor logits;
		Tensor dropout_keep_prob;

		
		//Path to a directory containing: saved folder, checkpoints folder, vocab_words and vocab_labels files
		String modelDir = "";
		
		String sentence = "beautiful like love peace best friends";
		
		
		String vocabWordsFile = modelDir + "vocab_words";
		String vocabLabelsFile = modelDir + "vocab_labels";
		int sequenceLength = 119;
		TextProcessor textProcessor = new TextProcessor(sequenceLength, vocabWordsFile, vocabLabelsFile);
		
		
		sentence = sentence.toLowerCase();
		int batch_size = 1;


		String[][] data = new String[batch_size][sequenceLength];
		data[0] = textProcessor.splitPadSentence(sentence);

		input_t = textProcessor.textToInputTensor(data);
		Float prob = (float) 1.0;
		dropout_keep_prob = Tensor.create(prob);

		String savedModelFolder = modelDir + "saved";
		SavedModelBundle model = SavedModelBundle.load(savedModelFolder, "serve");
		Session s = model.session();
		Graph g = model.graph();
		
		/*
		Iterator<Operation> it = g.operations();
		while(it.hasNext()) {
			Operation op = it.next();
			System.out.println("OP->" + op.name());
		}
		*/

		output = s
				.runner()
				.feed("input_x", input_t)
				.feed("dropout_keep_prob", dropout_keep_prob)
				.fetch("output/predictions")
				.fetch("output/logits")
				.run();


		prediction = output.get(0);
		System.out.println("Output prediction ----->" + prediction);


		int[] copy_prediction = new int[batch_size];
		prediction.copyTo(copy_prediction);
		for (int i = 0; i < batch_size; i++) {
			int int_pred = copy_prediction[i];
			String string_pred = textProcessor.labelToString(int_pred);
			System.out.println("LABEL_ID: " + int_pred + " -> " + string_pred);
		}


		logits = output.get(1);
		System.out.println("Output logits ----->" + logits);

		int nlabels = (int) logits.shape()[1];


		float[][] copy_logits = new float[batch_size][nlabels];
		logits.copyTo(copy_logits);
		for (int i = 0; i < nlabels; i++) {
			System.out.println(copy_logits[0][i]);

		}

	}




	private static byte[] readAllBytesOrExit(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(1);
		}
		return null;
	}

	/*
	  private static void printSignature(SavedModelBundle model) throws Exception {
		    MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
		    SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
		    int numInputs = sig.getInputsCount();
		    int i = 1;
		    System.out.println("MODEL SIGNATURE");
		    System.out.println("Inputs:");
		    for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
		      TensorInfo t = entry.getValue();
		      System.out.printf(
		          "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
		          i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
		    }
		    int numOutputs = sig.getOutputsCount();
		    i = 1;
		    System.out.println("Outputs:");
		    for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
		      TensorInfo t = entry.getValue();
		      System.out.printf(
		          "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
		          i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
		    }
		    System.out.println("-----------------------------------------------");
		  }
	 */

}
