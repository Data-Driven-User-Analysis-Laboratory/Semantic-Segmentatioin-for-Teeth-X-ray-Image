 Annotations - Object Detection에 필요한 xml 라벨 데이터가 포함된 폴더

 ImageSets - Action, Layout, Main, Segmentation 네 가지 폴더로 나눠져있으며, 각 폴더마다 txt 파일이 들어가있음.
인터넷 검색 결과 어떤 이미지 그룹을 test, train, trainval, val 로 사용할 것인지, 특정 클래스가 어떤 이미지에 있는지 등에 대한
정보를 포함하고 있는 폴더
* Action : Unknwon
* Layout : Unknwon
* Main : Unknwon
* Segmentation : 어떤 이미지로 Segmentation을 진행할지에 대한 라벨링 데이터

 JPEGImages - 원본 이미지가 들어가있는 폴더. Object Detection에서 input data가 된다고 함.
 
 SegmentationClass - semantic segmentation에 필요한 mask 데이터가 포함된 폴더

 SegmentationObject - Instance segmentation에 필요한 mask 데이터가 포함된 폴더