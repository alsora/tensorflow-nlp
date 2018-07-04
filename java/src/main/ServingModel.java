package main;

import serving_utils.SequenceTaggingModel;

public class ServingModel {

	
	public static void main(String[] args) throws Exception {
	
		String modelPath = "";
		
		SequenceTaggingModel model = new SequenceTaggingModel(modelPath);
		
		String data = "I love London and New York";
		
		String[] result = model.inference(data);
	
		for (int i = 0; i < result.length; i++) {
			String input = data.split(" ")[i];
			String res = result[i];
			System.out.println(input + "-->" + res);
		}
		

	}
	
	
}
