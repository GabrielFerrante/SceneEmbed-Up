## Download 250GB COYO


1. Change paths if you need.


img2dataset --url_list F:/COYO/coyo_meta/data --input_format "parquet" --url_col "url" --caption_col "text" --output_format webdataset --output_folder F:/COYO/coyo --processes_count 16 --thread_count 64 --image_size 256 --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True --number_sample 15000000
             
Stop in 15 million counts