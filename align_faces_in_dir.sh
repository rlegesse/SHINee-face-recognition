#! /bin/bash


for file in ~/projects/SHINee_face_recognition/scraped_data/minho/*; do
	basename "$file"
	f="$(basename -- $file)"
	#echo $file
	#echo "~/Desktop/$f"
	python align_faces.py --shape-predictor "shape_predictor_68_face_landmarks.dat" --image $file -o ~/Desktop/minho_aligned/ -n $f
done

