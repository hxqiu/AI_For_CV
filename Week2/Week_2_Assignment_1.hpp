#include <string>
#include <vector>

using namespace std;
typedef std::vector<std::vector<int>> Image;
int getMedian(vector<int> &hist, int kernal_side) {
	int cur_nums = 0;
	if (kernal_side % 2 == 1) {
		int mid = kernal_side * kernal_side / 2;
		for (int i = 0; i < hist.size(); i++) {
			if (cur_nums < mid)
				cur_nums += hist[i];
			if (cur_nums > mid)
				return hist[i];
		}
	}
	else {
		int mid = kernal_side * kernal_side / 2;
		int l_bound, r_bound, i;
		for (i = 0; i < hist.size(); i++) {
			if (cur_nums < mid)
				cur_nums += hist[i];
			if (cur_nums >= mid)
				l_bound = i;
		}
		for (; i < hist.size(); i++) {
			mid++;
			if (cur_nums < mid)
				cur_nums += hist[i];
			if (cur_nums >= mid)
				r_bound = i;
		}
		return (hist[l_bound] + hist[r_bound]) / 2;
	}
}
void init_hist(vector<vector<int>>& img, vector<vector<int>> kernel, vector<int> &hist, int cur_h) {
	int h = img.size();
	int w = img[0].size();
	int k_h = kernel.size();
	int k_w = kernel[0].size();
	for (int i = cur_h - k_h / 2; i <= cur_h + k_h / 2; i++) {
		for (int j = 0; j < k_w; j++) {
			hist[img[i][j]]++;
		}
	}
}
void medianBlur(vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way) {
	vector<vector<int>> img_padding(img.size() + 2, vector<int>(img[0].size() + 2, 0));
	if (padding_way == "same") {
		//填充图像周围kernal.size()/2宽度的相同像素
		for (int i = 0; i < img_padding.size() - (kernel.size() / 2) * 2; i++) {
			for (int j = 0; j < kernel[0].size() / 2; j++) {
				img_padding[i][j] = img[i][0];
				img_padding[i][img_padding[0].size() - j - 1] = img[i][img[0].size() - 1];
			}
		}
		for (int i = 0; i < kernel.size() / 2; i++) {
			for (int j = 0; j < img_padding.size() - (kernel[0].size() / 2) * 2; j++) {
				img_padding[i][j] = img[0][j];
				img_padding[img_padding.size() - i - 1][j] = img[img.size() - 1][j];
			}
		}
		for (int i = 0; i < kernel.size() / 2; i++) {
			for (int j = 0; j < kernel[0].size() / 2; j++) {
				img_padding[i][j] = img[0][0];
				img_padding[img_padding.size() - i - 1][j] = img[img.size() - 1][0];
				img_padding[0][img_padding.size() - j - 1] = img[0][img[0].size() - 1];
				img_padding[img_padding.size() - i - 1][img_padding.size() - j - 1] = img[img.size() - 1][img[0].size() - 1];
			}
		}
	}
	else
		img_padding = img;

	int h = img_padding.size();
	int w = img_padding[0].size();
	int k_h = kernel.size();
	int k_w = kernel[0].size();
	vector<int> hist(256, 0);//histogram[256]
	Image ret_map(h - 2, vector<int>(w - 2, 0)); //result_feature_map

	for (int i = 1; i < h - 1; i++) {//no padding 
		init_hist(img_padding, kernel, hist, i);
		ret_map[i - 1][0] = getMedian(hist, kernel.size());
		for (int j = 2; j < w - 1; j++) {// no padding
			for (int n = 0; n < k_h; n++) {
				hist[img_padding[i - k_h / 2 + n][j - k_w / 2 - 1]]--;//减去最左侧一列
				hist[img_padding[i - k_h / 2 + n][j + k_w / 2]]++;//加上最右侧一列
			}
			ret_map[i - 1][j - 2] = getMedian(hist, kernel.size());
		}
	}
}