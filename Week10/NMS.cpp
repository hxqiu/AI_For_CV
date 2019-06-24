#include "pch.h"
#include <vector>
#include <algorithm>

using namespace std;

class BoundingBox {
public:
	float x;
	float y;
	float w;
	float h;

	float area() {
		return w * h;
	}

	BoundingBox() {
		x = y = w = h = 0.0;
	}

	BoundingBox(float x, float y, float w, float h) {
		this->x = x;
		this->y = y;
		this->w = w;
		this->h = h;
	}
};

float cal_IoU(BoundingBox& BoxA,
	BoundingBox& BoxB) {
	float intersection_x = min(BoxA.x + BoxA.w, BoxB.x + BoxB.w) - max(BoxA.x, BoxB.x);
	float intersection_y = min(BoxA.y + BoxA.h, BoxB.y + BoxB.h) - max(BoxA.y, BoxB.y);
	if (intersection_x < 0.0 || intersection_y < 0.0)
		return 0.0;
	float area_of_intersection = intersection_x * intersection_y;
	float area_of_union = BoxA.area() + BoxB.area() - area_of_intersection;
	return area_of_intersection / area_of_union;
}

vector<vector<float>>NMS(vector<vector<float>>lists, float thre) {
	vector<BoundingBox> B;
	vector<float> S;
	vector<BoundingBox> D;
	vector<float> result_S;
	for (vector<float> list : lists) {
		B.push_back(BoundingBox(list[0], list[1], list[2], list[3]));
		S.push_back(list[4]);
	}
	BoundingBox M;
	while (!B.empty()) {
		auto max_element_it = max_element(S.begin(), S.end());
		int max_element_ind = max_element_it - S.begin();
		M = B[max_element_ind];
		D.push_back(M);
		result_S.push_back(S[max_element_ind]);
		B.erase(B.begin() + max_element_ind);
		S.erase(S.begin() + max_element_ind);
		for (int i = 0; i < B.size(); i++) {
			if (cal_IoU(B[i], M) > thre) {
				B.erase(B.begin() + i);
				S.erase(S.begin() + i);
				--i;
			}
		}
	}
	vector<vector<float>> ret;
	for (int i = 0; i < D.size(); i++) {
		ret.push_back({ D[i].x, D[i].y, D[i].w, D[i].h, result_S[i] });
	}
	return ret;
}

int main() {
	vector<float> list1 = { 254.0, 259.0, 205.0, 205.0, 0.87 };
	vector<float> list2 = { 102.0, 189.0, 285.0, 285.0, 0.9 };
	vector<float> list3 = { 635.0, 372.0, 463.0, 743.0, 0.72 };
	vector<float> list4 = { 643.0, 243.0, 532.0, 634.0, 0.2 };
	vector<vector<float>> lists;
	lists.push_back(list1);
	lists.push_back(list2);
	lists.push_back(list3);
	lists.push_back(list4);
	vector<vector<float>> result = NMS(lists, 0.3);
	return 0;
}