package data_utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.Tensor;

public class TextProcessor {
	
    private Map<String,Integer> vocabWordsMap = new HashMap<String, Integer>();
    private Map<Integer,String> vocabLabelsMap = new HashMap<Integer, String>();

	private int max_item_length;
	private String paddingToken = "<padding>";
	private String unknownToken = "<unk>";

	
	public TextProcessor(int sequenceLength, String wordsPath, String labelsPath) {
		
		this.max_item_length = sequenceLength;
		
		this.vocabWordsMap = loadVocabMap(wordsPath);
		this.vocabLabelsMap = loadReversedVocabMap(labelsPath);
		
	}
	
	
	public Map<String,Integer> loadVocabMap(String path) {
		
		Map<String,Integer> vocabMap = new HashMap<String, Integer>();
		
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
		}
		
		
		return vocabMap;
			
	}
	
	public Map<Integer, String> loadReversedVocabMap(String path){
		
		Map<Integer,String> vocabMap = new HashMap<Integer, String>();
		
		try {
	        File file = new File(path);
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line;
			int id = 0;
			while ((line = bufferedReader.readLine()) != null) {
				line = line.trim();
				vocabMap.put(id, line);
				id ++;
			}
			fileReader.close();
		}
		catch (IOException ex) {
			System.out.println("File not found: " + path);
		}
		
		
		return vocabMap;
		
		
	}
	
	
	public String[] padSentence(String[] sentence) {
		
		
		String[] output = new String[this.max_item_length]; 
		
		for (int i = 0; i < this.max_item_length; i++) {
			if (i < sentence.length) {
				output[i] = sentence[i];
			}
			else {
				output[i] = paddingToken;
			}
		}
		
		return output;
		
	}
	
	
	public String[] splitPadSentence(String sentence) {
		
		String[] splitted = sentence.split(" ");
		
		String[] padded = padSentence(splitted);
		
		return padded;

	}
	
	
	public Tensor textToInputTensor(String[][] data) {
		
		if (data.length == 0) {
			System.out.println("ERROR: provided empty data to TextProcessor.textToInputTensor!!!");
			return Tensor.create(new int[1][this.max_item_length]);
		}
		
		int[][] matrix_batch= new int[data.length][data[0].length];
		
		for (int b = 0; b < data.length; b ++) {
			
			if (data[b].length != this.max_item_length) {
				System.out.println("ERROR: element of data has wrong length in TextProcessor.textToInputTensor!!!");
				return Tensor.create(new int[1][this.max_item_length]);
			}
			
			for (int s = 0; s < data[b].length; s ++) {
				
				String word = data[b][s];
				
				int word_id;
				
			   if (vocabWordsMap.containsKey(word)) {
				   word_id = vocabWordsMap.get(word);
				   } 
			   else {
				   word_id = vocabWordsMap.get(unknownToken);
			    }
			   
			   matrix_batch[b][s] = word_id;
				  	   
			}
			
		}
		
		
		return Tensor.create(matrix_batch);
		
	}
	
	public String labelToString(int labelID) {
		
		 if (vocabLabelsMap.containsKey(labelID)) {
			 return vocabLabelsMap.get(labelID);
		 }
		 else {
			 System.out.println("Label not found: " + labelID);
			 return "";
		 }
		
	}
	
	
	

}
