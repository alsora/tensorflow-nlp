package data_utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.Tensor;

public class TextProcessor {
	
    private Map<String,Integer> vocabMap = new HashMap<String, Integer>();
	private int max_item_length = 20;
	private String paddingToken = "<padding>";
	private String unknownToken = "<unk>";

	
	public TextProcessor() {
		
		
		
	}
	
	
	public void loadVocabMap(String path) {
		
		vocabMap.clear();
		
		try {
	        File file = new File(path);
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line;
			int id = 0;
			while ((line = bufferedReader.readLine()) != null) {
				line = line.trim();
				vocabMap.put(line, id);
				id ++;
			}
			fileReader.close();
		}
		catch (IOException ex) {
			System.out.println("File not found: " + path);
			vocabMap.put(unknownToken, 0); //Ensure that at least the unknown token is present in the VocabMap
		}
			
	}
	
	
	public String[] splitPadSentence(String sentence) {
		
		String[] splitted = sentence.split(" ");
		String[] output = new String[max_item_length]; 
		
		
		for (int i = 0; i < max_item_length; i++) {
			String word;
			if (i < splitted.length) {
				output[i] = splitted[i];
			}
			else {
				output[i] = paddingToken;
			}
		}
		
		
		return output;
		
	}
	
	
	public Tensor textToInputTensor(String[] splittedSentence) {
		
		int batch_size = 1;
		
		int[][] matrix_batch= new int[batch_size][splittedSentence.length];
		
		
		for (int i = 0; i < splittedSentence.length; i++) {
			
			String word = splittedSentence[i];

			int word_id;

				
		   if (vocabMap.containsKey(word)) {
			   word_id = vocabMap.get(word);
			   } 
		   else {
			   word_id = vocabMap.get(unknownToken);
		    }
		

			for (int j = 0; j < batch_size; j ++) {
				matrix_batch[j][i] = word_id;
			}
			
		}
		
		
		return Tensor.create(matrix_batch);
		
	}
	
	
	
	
	

}
