## Download 15 millions images and captions

1. Change paths if you need.


img2dataset --url_list F:/COYO/coyo_500gb_meta/data --input_format "parquet" --url_col "url" --caption_col "text"  --output_format webdataset --output_folder F:/COYO/coyo_500gb  --processes_count 16 --thread_count 64 --image_size 256 --resize_mode keep_ratio --number_sample 15000000

img2dataset --url_list F:/COYO/coyo_500gb_meta/data --input_format "parquet" --url_col "url" --caption_col "text" --output_format webdataset --output_folder F:/COYO/coyo_500gb --processes_count 16 --thread_count 64 --image_size 256 --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True --number_sample 15000000
             