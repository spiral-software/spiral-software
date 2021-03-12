
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

sort_1 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 1, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_3_2_7_2_1_1_1_1");

end;

sort_2 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 1, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 1, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 1, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_9_8_7_6_5_4_3_2");

end;

sort_3 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 16, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_3_2_7_1_1_1_1_1");

end;

sort_4 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 16, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 16, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_16_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 16, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_16_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_9_4_7_3_5_2_3_1");

end;

sort_5 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 32, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_3_2_1_1_1_1_1_1");

end;

sort_6 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 32, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 32, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_32_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 32, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_32_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_3_4_7_2_5_2_1_1");

end;

sort_7 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 64, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 64, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_3_1_1_1_1_1_1_1");

end;

sort_8 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 64, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_64_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 64, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_64_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_3_2_7_2_5_1_1_1");

end;

sort_9 := function()
	HDLSynthesize_no_brams(sortAlg2(512, 128, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(512, 128, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_512_128_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(512, 128, [1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_512_128_1_1_1_1_1_1_1_1");

end;

sort_10 := function()
	HDLSynthesize_no_brams(sortAlg4(512, 128, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_512_128_1");
	HDLSynthesize_no_brams(sortAlg4(512, 128, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_512_128_3");
	HDLSynthesize_no_brams(sortAlg4(512, 128, 9), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_512_128_9");
	HDLSynthesize_no_brams(sortAlg4(512, 128, 27), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_512_128_27");
	HDLSynthesize_no_brams(sortAlg4(512, 128, 81), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_512_128_81");
	HDLSynthesize_no_brams(sortAlg6(1024, 1), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_1");
	HDLSynthesize_no_brams(sortAlg1(1024, 2), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_2");
	HDLSynthesize_no_brams(sortAlg6(1024, 2), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_2");
	HDLSynthesize_no_brams(sortAlg1(1024, 4), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_4");
	HDLSynthesize_no_brams(sortAlg6(1024, 4), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_4");

end;

sort_11 := function()
	HDLSynthesize_no_brams(sortAlg1(1024, 8), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_8");
	HDLSynthesize_no_brams(sortAlg6(1024, 8), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_8");
	HDLSynthesize_no_brams(sortAlg1(1024, 16), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_16");
	HDLSynthesize_no_brams(sortAlg6(1024, 16), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_16");
	HDLSynthesize_no_brams(sortAlg1(1024, 32), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_32");
	HDLSynthesize_no_brams(sortAlg6(1024, 32), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_32");
	HDLSynthesize_no_brams(sortAlg1(1024, 64), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_64");
	HDLSynthesize_no_brams(sortAlg6(1024, 64), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_64");
	HDLSynthesize_no_brams(sortAlg1(1024, 128), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_1024_128");
	HDLSynthesize_no_brams(sortAlg6(1024, 128), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_1024_128");

end;

sort_12 := function()
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_2_3_2_7_2_1_1_1_1");

end;

sort_13 := function()
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 1, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 1, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_1_1_1_1_1_1_1_1_1_1");

end;

sort_14 := function()
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_2_3_2_7_2_1_1_1_1");

end;

sort_15 := function()
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 2, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_2_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 2, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_2_1_1_1_1_1_1_1_1_1");

end;

sort_16 := function()
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_2_3_2_7_2_1_1_1_1");

end;

sort_17 := function()
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 4, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_4_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 4, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_4_1_1_1_1_1_1_1_1_1");

end;

sort_18 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 16, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_16_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 16, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_16_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 16, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_16_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 16, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_16_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 16, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_16_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_5_3_4_7_2_5_2_1_1");

end;

sort_19 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_2_3_1_1_1_1_1_1_1");

end;

sort_20 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 32, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_32_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 32, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_32_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_5_3_4_7_2_5_2_1_1");

end;

sort_21 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_2_3_1_1_1_1_1_1_1");

end;

sort_22 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 64, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_64_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 64, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_64_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_5_3_4_7_2_5_2_1_1");

end;

sort_23 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_2_3_1_1_1_1_1_1_1");

end;

sort_24 := function()
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(1024, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_1024_128_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(1024, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_1024_128_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_1");
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_2");
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 5), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_5");
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 10), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_10");
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 20), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_20");

end;

sort_25 := function()
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 50), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_50");
	HDLSynthesize_no_brams(sortAlg4(1024, 16, 100), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_16_100");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_1");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_2");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 5), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_5");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 10), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_10");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 20), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_20");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 50), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_50");
	HDLSynthesize_no_brams(sortAlg4(1024, 32, 100), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_32_100");
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_1");

end;

sort_26 := function()
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_2");
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 5), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_5");
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 10), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_10");
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 20), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_20");
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 50), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_50");
	HDLSynthesize_no_brams(sortAlg4(1024, 64, 100), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_64_100");
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_1");
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_2");
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 5), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_5");
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 10), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_10");

end;

sort_27 := function()
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 20), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_20");
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 50), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_50");
	HDLSynthesize_no_brams(sortAlg4(1024, 128, 100), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_1024_128_100");
	HDLSynthesize_no_brams(sortAlg5(2048, 16, [11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_16_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 16, [11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_16_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 16, [11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_16_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 16, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_16_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 16, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_16_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_10_9_8_7_6_5_4_3_2");

end;

sort_28 := function()
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_2_3_2_7_1_1_1_1_1");

end;

sort_29 := function()
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 32, [11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_32_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 32, [11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_32_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 128, [11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_128_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 128, [11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_128_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 128, [11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_128_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_11_2_3_2_7_2_1_1_1_1");

end;

sort_30 := function()
	HDLSynthesize_no_brams(sortAlg5(2048, 128, [11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_128_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 128, [11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_128_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(2048, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_2048_128_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(2048, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_2048_128_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg4(2048, 1, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_1_1");
	HDLSynthesize_no_brams(sortAlg4(2048, 1, 11), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_1_11");
	HDLSynthesize_no_brams(sortAlg4(2048, 1, 121), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_1_121");
	HDLSynthesize_no_brams(sortAlg4(2048, 2, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_2_1");

end;

sort_31 := function()
	HDLSynthesize_no_brams(sortAlg4(2048, 2, 11), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_2_11");
	HDLSynthesize_no_brams(sortAlg4(2048, 2, 121), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_2_121");
	HDLSynthesize_no_brams(sortAlg4(2048, 4, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_2048_4_1");
	HDLSynthesize_no_brams(sortAlg6(4096, 8), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_4096_8");
	HDLSynthesize_no_brams(sortAlg1(4096, 16), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_4096_16");
	HDLSynthesize_no_brams(sortAlg6(4096, 16), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_4096_16");
	HDLSynthesize_no_brams(sortAlg1(4096, 32), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_4096_32");
	HDLSynthesize_no_brams(sortAlg6(4096, 32), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_4096_32");
	HDLSynthesize_no_brams(sortAlg1(4096, 64), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_4096_64");
	HDLSynthesize_no_brams(sortAlg6(4096, 64), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_4096_64");

end;

sort_32 := function()
	HDLSynthesize_no_brams(sortAlg1(4096, 128), 1, 0, 16, 350, 1,"sortAlg1_noBRAMs_4096_128");
	HDLSynthesize_no_brams(sortAlg6(4096, 128), 1, 0, 16, 350, 1,"sortAlg6_noBRAMs_4096_128");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_3_11_5_3_2_7_2_5_1_1_1");

end;

sort_33 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_2_11_2_1_1_1_1_1_1_1_1");

end;

sort_34 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_1_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_1_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_6_11_5_9_4_7_3_5_2_3_1");

end;

sort_35 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_2_11_2_3_2_1_1_1_1_1_1");

end;

sort_36 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_2_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_2_1_1_1_1_1_1_1_1_1_1_1");

end;

sort_37 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_3_11_2_3_2_7_2_1_1_1_1");

end;

sort_38 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_2_11_1_1_1_1_1_1_1_1_1");

end;

sort_39 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 4, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_4_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 4, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_4_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_4_11_5_3_4_7_2_5_2_1_1");

end;

sort_40 := function()
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 8, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_8_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 8, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_8_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_12_11_10_9_8_7_6_5_4_3_2");

end;

sort_41 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_2_11_2_3_2_7_1_1_1_1_1");

end;

sort_42 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_2_1_1_1_1_1_1_1_1_1_1");

end;

sort_43 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 16, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_16_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 16, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_16_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_3_11_5_3_2_7_2_5_1_1_1");

end;

sort_44 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_2_11_2_1_1_1_1_1_1_1_1");

end;

sort_45 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 32, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_32_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 32, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_32_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_6_11_5_9_4_7_3_5_2_3_1");

end;

sort_46 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_2_11_2_3_2_1_1_1_1_1_1");

end;

sort_47 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 64, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_64_1_1_1_1_1_1_1_1_1_1_1");

end;

sort_48 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 64, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_64_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_12_11_10_9_8_7_6_5_4_3_2");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [6, 11, 5, 9, 4, 7, 3, 5, 2, 3, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_6_11_5_9_4_7_3_5_2_3_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [4, 11, 5, 3, 4, 7, 2, 5, 2, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_4_11_5_3_4_7_2_5_2_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [3, 11, 5, 3, 2, 7, 2, 5, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_3_11_5_3_2_7_2_5_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_3_11_2_3_2_7_2_1_1_1_1");

end;

sort_49 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [3, 11, 2, 3, 2, 7, 2, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_3_11_2_3_2_7_2_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [2, 11, 2, 3, 2, 7, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_2_11_2_3_2_7_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [2, 11, 2, 3, 2, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_2_11_2_3_2_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [2, 11, 2, 3, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_2_11_2_3_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [2, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_2_11_2_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [2, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_2_11_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_2_1_1_1_1_1_1_1_1_1_1");

end;

sort_50 := function()
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_2_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg2(4096, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg2_noBRAMs_4096_128_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg5(4096, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1, 0, 16, 350, 1,"sortAlg5_noBRAMs_4096_128_1_1_1_1_1_1_1_1_1_1_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_24");

end;

sort_51 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 1, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_1_144");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_12");

end;

sort_52 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 2, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_2_144");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_6");

end;

sort_53 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 4, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_4_144");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_4");

end;

sort_54 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 8, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_8_144");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_3");

end;

sort_55 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 16, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_16_144");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_2");

end;

sort_56 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 32, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_32_144");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_1");

end;

sort_57 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 64, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_64_144");

end;

sort_58 := function()
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 1), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_1");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 2), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_2");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 3), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_3");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 4), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_4");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 6), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_6");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 12), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_12");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 24), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_24");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 36), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_36");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 48), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_48");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 72), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_72");
	HDLSynthesize_no_brams(sortAlg4(4096, 128, 144), 1, 0, 16, 350, 1,"sortAlg4_noBRAMs_4096_128_144");
end;
