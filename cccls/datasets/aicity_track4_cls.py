import numpy as np
from collections import defaultdict

from mmcls.datasets import DATASETS, BaseDataset

@DATASETS.register_module()
class AICIty22Track4ClsDataset(BaseDataset):

    CLASSES = (
        "Advil Liquid Gel",
        "Advil Tablets",
        "Alcohol Pads",
        "Aleve PM",
        "Aleve Tablets",
        "All Free and Clear Pods",
        "Arm and Hammer Toothpaste",
        "Bandaid Variety",
        "Banza",
        "Barilla Penne",
        "Barilla Red Lentil",
        "Barilla Rotini Pasta",
        "Barnums Animal Crackers",
        "Bayer Chewable 3 Pack",
        "Bayer Lower Dose",
        "Benadryl Allergy Tablets",
        "Biore",
        "Brown Rice",
        "Bumble Bee Tuna",
        "Cane Sugar",
        "Chamomile Tea",
        "Chicken Bouillon",
        "Children_s Allegra Allergy",
        "Children_s Zyrtec",
        "Chocolate Fudge Poptart",
        "Chocolate Pocky",
        "Chocolate Strawberries",
        "Claratin Chewable",
        "Coffee Mate Creamer",
        "Crackers and Cheese",
        "Cream Cheese",
        "Crest White Strips",
        "Dove Cool Moisturizer",
        "Dove Soap Sensitive",
        "Dove Body Wash",
        "Downy Odor Defense",
        "Dreft Scent Beads",
        "Dry Eye Relief",
        "Excedrine Migraine",
        "Extra Spearmint",
        "Flonase",
        "Flow Flex Covid 19 Test",
        "French Roast Starbucks",
        "Frosting Mix Chocolate",
        "Fudge Oreos",
        "Funfetti Cake",
        "Gain Fireworks Scent Beads",
        "Gain Flings",
        "Garnier Shampoo",
        "Glad Drawstring Trash Bag",
        "Gluten Free Brownies",
        "Gluten Free Pasta",
        "Gournay Cheese",
        "Healthy Choice Chicken Alfredo",
        "Heavy Duty Forks",
        "Hefty Small Trash Bags",
        "Hello Toothpaste",
        "Honey Maid Graham Crackers",
        "Ice Cream Sandwiches",
        "Ice Breakers Ice Cubes",
        "Jello Vanilla Pudding",
        "Kerrygold Butter",
        "Kleenex",
        "Lipton Noodle Soup",
        "Mac 'n' Cheese Shells",
        "Macaroni and Cheese Original",
        "Matzo Balls",
        "McCormick Natures Food Coloring",
        "Milk Duds",
        "Minute Tapioca",
        "Miss Dior",
        "Mixed Berries Hibiscus Tea",
        "M&Ms Peanuts",
        "Mochi Ice Cream",
        "Motrin Ib Migraine",
        "Mr Clean",
        "Nasacort Allergy",
        "Nasal Decongestant",
        "Nasal Strip",
        "Nature Valley Granola Bars",
        "Nintendo Switch Controllers",
        "Onion Soup Dip Mix",
        "Pedialyte Powder",
        "Peets Keurig",
        "Peet's Coffee Grounds Dark Roast",
        "Playtex Sport",
        "Pocky Strawberry",
        "Pork Sausage",
        "Pure Vanilla Extract",
        "Raisinets",
        "Ranch Seasoning",
        "Reese's Pieces",
        "Rewetting Drops Contact Solution",
        "Smores Poptarts",
        "Snuggle Dryer Sheets",
        "Sour Patch Kids",
        "Stevia in the Raw",
        "Strawberry Jello",
        "Stress Relief Spray All Natural",
        "Sunmaid Raisins",
        "Swiss Miss Hot Chocolate",
        "Tampax",
        "Tide Pods",
        "Toothpicks",
        "Tostada Shells",
        "Total Home Scent Boos",
        "Tussin",
        "Tylenol Arthritis",
        "Unstoppables",
        "Vapo Rub",
        "Vick's Pure Zzz's",
        "Vinyl Gloves",
        "Visine Red Eye Hydrating Comfort",
        "Whoppers",
        "Wild Strawberry Tea",
        "Woolite Delicates _Attempt_",
    )

    def load_annotations(self):
        filenames = np.loadtxt(self.ann_file, dtype=str)

        data_infos = []
        self.class_imgs = defaultdict(lambda: list())

        for filename in filenames:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            gt_label = int(filename.split('_')[0]) - 1  # base1 --> base0
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)

            img_file = f"{self.data_prefix}/{filename}"
            self.class_imgs[gt_label].append(img_file)

        print(f"[{self.__class__.__name__}] "
              f"Number of samples: {len(data_infos)}")
        return data_infos

    def query_imgs(self, cls_idx, num_imgs=1):
        img_files = self.class_imgs[cls_idx]
        inds = np.random.randint(0, len(img_files), size=(num_imgs, ))
        selected_img_files = [img_files[idx] for idx in inds]
        return selected_img_files
