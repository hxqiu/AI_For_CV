#include "pch.h"
#include <vector>
#include <algorithm>

using namespace std;

class BoundingBox {
public:
	double x;
	double y;
	double w;
	double h;

	double area() {
		return w * h;
	}
};

double cal_IoU(BoundingBox& BoxA,
	           BoundingBox& BoxB) {
	double intersection_x = min(BoxA.x + BoxA.w, BoxA.x + BoxB.w) - max(BoxA.x, BoxB.x);
	double intersection_y = min(BoxA.y + BoxA.h, BoxB.y + BoxB.h) - max(BoxA.y, BoxB.y);
	if (intersection_x < 0.0 || intersection_y < 0.0)
		return 0.0;
	double area_of_intersection = intersection_x * intersection_y;
	double area_of_union = BoxA.area + BoxB.area - area_of_intersection;
	return area_of_intersection/ area_of_union;
}
void NMS(vector<BoundingBox> &B, 
		 vector<double> &S,
		 double threshold,
	     vector<BoundingBox> &D) {
	BoundingBox M;
	while (!B.empty()) {
		auto max_element_it = max_element(S.begin(), S.end());
		int max_element_ind = max_element_it - S.begin();
		M = B[max_element_ind];
		D.push_back(M);
		B.erase(B.begin() + max_element_ind);
		for (int i = 0; i < B.size(); i++) {
			if (cal_IoU(B[i], M) > threshold) {
				B.erase(B.begin() + i);
				S.erase(S.begin() + i);
				--i;
			}
		}
	}	
}
int main() {
	BoundingBox b1(254.0, 259.0, 205.0, 205.0);
	BoundingBox b2(102.0, 189.0, 285.0, 285.0);
	BoundingBox b3(635.0, 372.0, 463.0, 743.0);
	BoundingBox b4(643.0, 243.0, 532.0, 634.0);
	
	vector<BoundingBox> list_bbox;
	vector<double> scores = { 0.87, 0.9, 0.72, 0.2 };
	vector<BoundingBox> result_b;
	vector<double> result_scores;
	list_bbox.push_back(b1);
	list_bbox.push_back(b2);
	list_bbox.push_back(b3);
	list_bbox.push_back(b4);
	NMS(list_bbox, scores, 0.3, result_b, result_scores);
	return 0;
}