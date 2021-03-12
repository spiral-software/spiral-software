
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

2ddft_1 := function()
	HDLSynthesize(stream2DDFT(4, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT4_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(4, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT4_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(4, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT4_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(4, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT4_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(4, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT4_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(4, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT4_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(4, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT4_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(4, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT4_4_4_1_2_16bits");

end;

2ddft_2 := function()
	HDLSynthesize(stream2DDFT(8, 2, 4, 1, 3), 1, 0, 16, 380, 1,"2DDFT8_2_2_3_1_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 4, 2, 3), 1, 0, 16, 380, 1,"2DDFT8_2_2_3_2_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT8_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT8_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 8, 1, 3), 1, 0, 16, 380, 1,"2DDFT8_2_4_3_1_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 8, 2, 3), 1, 0, 16, 380, 1,"2DDFT8_2_4_3_2_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT8_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT8_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT8_2_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT8_2_8_3_2_16bits");

end;

2ddft_3 := function()
	HDLSynthesize(stream2DDFT(8, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT8_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(8, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT8_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(8, 8, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT8_8_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(8, 8, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT8_8_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 4, 1, 4), 1, 0, 16, 380, 1,"2DDFT16_2_2_4_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 4, 2, 4), 1, 0, 16, 380, 1,"2DDFT16_2_2_4_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 4, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_2_2_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 4, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_2_2_2_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_2_2_1_2_16bits");

end;

2ddft_4 := function()
	HDLSynthesize(stream2DDFT(16, 2, 8, 1, 4), 1, 0, 16, 380, 1,"2DDFT16_2_4_4_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 8, 2, 4), 1, 0, 16, 380, 1,"2DDFT16_2_4_4_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 16, 1, 4), 1, 0, 16, 380, 1,"2DDFT16_2_8_4_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 16, 2, 4), 1, 0, 16, 380, 1,"2DDFT16_2_8_4_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_2_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_2_8_2_2_16bits");

end;

2ddft_5 := function()
	HDLSynthesize(stream2DDFT(16, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 32, 1, 4), 1, 0, 16, 380, 1,"2DDFT16_2_16_4_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 32, 2, 4), 1, 0, 16, 380, 1,"2DDFT16_2_16_4_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_2_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_2_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_4_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_4_4_2_2_16bits");

end;

2ddft_6 := function()
	HDLSynthesize(stream2DDFT(16, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_4_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_4_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_4_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_4_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_4_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT16_4_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT16_4_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_4_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 4, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_4_16_1_2_16bits");

end;

2ddft_7 := function()
	HDLSynthesize(stream2DDFT(16, 16, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT16_16_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(16, 16, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT16_16_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 4, 1, 5), 1, 0, 16, 380, 1,"2DDFT32_2_2_5_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 4, 2, 5), 1, 0, 16, 380, 1,"2DDFT32_2_2_5_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT32_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT32_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 8, 1, 5), 1, 0, 16, 380, 1,"2DDFT32_2_4_5_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 8, 2, 5), 1, 0, 16, 380, 1,"2DDFT32_2_4_5_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT32_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT32_2_4_1_2_16bits");

end;

2ddft_8 := function()
	HDLSynthesize(stream2DDFT(32, 2, 16, 1, 5), 1, 0, 16, 380, 1,"2DDFT32_2_8_5_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 16, 2, 5), 1, 0, 16, 380, 1,"2DDFT32_2_8_5_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT32_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT32_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 32, 1, 5), 1, 0, 16, 380, 1,"2DDFT32_2_16_5_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 32, 2, 5), 1, 0, 16, 380, 1,"2DDFT32_2_16_5_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT32_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT32_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 64, 1, 5), 1, 0, 16, 380, 1,"2DDFT32_2_32_5_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 64, 2, 5), 1, 0, 16, 380, 1,"2DDFT32_2_32_5_2_16bits");

end;

2ddft_9 := function()
	HDLSynthesize(stream2DDFT(32, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT32_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(32, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT32_2_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(32, 32, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT32_32_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(32, 32, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT32_32_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 1, 6), 1, 0, 16, 380, 1,"2DDFT64_2_2_6_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 2, 6), 1, 0, 16, 380, 1,"2DDFT64_2_2_6_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_2_2_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_2_2_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_2_2_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_2_2_2_2_16bits");

end;

2ddft_10 := function()
	HDLSynthesize(stream2DDFT(64, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 1, 6), 1, 0, 16, 380, 1,"2DDFT64_2_4_6_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 2, 6), 1, 0, 16, 380, 1,"2DDFT64_2_4_6_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_2_4_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_2_4_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_2_4_1_2_16bits");

end;

2ddft_11 := function()
	HDLSynthesize(stream2DDFT(64, 2, 16, 1, 6), 1, 0, 16, 380, 1,"2DDFT64_2_8_6_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 2, 6), 1, 0, 16, 380, 1,"2DDFT64_2_8_6_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_2_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_2_8_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_2_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_2_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 1, 6), 1, 0, 16, 380, 1,"2DDFT64_2_16_6_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 2, 6), 1, 0, 16, 380, 1,"2DDFT64_2_16_6_2_16bits");

end;

2ddft_12 := function()
	HDLSynthesize(stream2DDFT(64, 2, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_2_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_2_16_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_2_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_2_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 1, 6), 1, 0, 16, 380, 1,"2DDFT64_2_32_6_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 2, 6), 1, 0, 16, 380, 1,"2DDFT64_2_32_6_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_2_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_2_32_3_2_16bits");

end;

2ddft_13 := function()
	HDLSynthesize(stream2DDFT(64, 2, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_2_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_2_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_2_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 1, 6), 1, 0, 16, 380, 1,"2DDFT64_2_64_6_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 2, 6), 1, 0, 16, 380, 1,"2DDFT64_2_64_6_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_2_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_2_64_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_2_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_2_64_2_2_16bits");

end;

2ddft_14 := function()
	HDLSynthesize(stream2DDFT(64, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 8, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_4_4_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 8, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_4_4_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_4_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_4_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_4_8_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_4_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_4_8_1_2_16bits");

end;

2ddft_15 := function()
	HDLSynthesize(stream2DDFT(64, 4, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_4_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_4_16_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_4_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_4_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_4_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_4_32_3_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_4_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_4_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT64_4_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT64_4_64_3_2_16bits");

end;

2ddft_16 := function()
	HDLSynthesize(stream2DDFT(64, 4, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_4_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 4, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_4_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_8_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_8_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_8_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_8_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_8_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_8_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_8_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_8_16_1_2_16bits");

end;

2ddft_17 := function()
	HDLSynthesize(stream2DDFT(64, 8, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_8_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_8_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_8_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_8_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT64_8_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT64_8_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_8_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 8, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_8_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(64, 64, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT64_64_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(64, 64, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT64_64_64_1_2_16bits");

end;

2ddft_18 := function()
	HDLSynthesize(stream2DDFT(128, 2, 4, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_2_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 4, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_2_7_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 8, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_4_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 8, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_4_7_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 16, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_8_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 16, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_8_7_2_16bits");

end;

2ddft_19 := function()
	HDLSynthesize(stream2DDFT(128, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 32, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_16_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 32, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_16_7_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 64, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_32_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 64, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_32_7_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_32_1_2_16bits");

end;

2ddft_20 := function()
	HDLSynthesize(stream2DDFT(128, 2, 128, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_64_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 128, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_64_7_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 256, 1, 7), 1, 0, 16, 380, 1,"2DDFT128_2_128_7_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 256, 2, 7), 1, 0, 16, 380, 1,"2DDFT128_2_128_7_2_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(128, 128, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT128_128_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(128, 128, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT128_128_128_1_2_16bits");

end;

2ddft_21 := function()
	HDLSynthesize(stream2DDFT(256, 2, 4, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_2_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_2_8_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_2_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_2_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_2_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_2_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_4_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_4_8_2_16bits");

end;

2ddft_22 := function()
	HDLSynthesize(stream2DDFT(256, 2, 8, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_4_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_4_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_8_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_8_8_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_8_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_8_4_2_16bits");

end;

2ddft_23 := function()
	HDLSynthesize(stream2DDFT(256, 2, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_16_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_16_8_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_16_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_16_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_16_2_2_16bits");

end;

2ddft_24 := function()
	HDLSynthesize(stream2DDFT(256, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_32_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_32_8_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_32_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_32_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_32_1_2_16bits");

end;

2ddft_25 := function()
	HDLSynthesize(stream2DDFT(256, 2, 128, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_64_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_64_8_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_64_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_64_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_128_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_128_8_2_16bits");

end;

2ddft_26 := function()
	HDLSynthesize(stream2DDFT(256, 2, 256, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_128_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_128_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 1, 8), 1, 0, 16, 380, 1,"2DDFT256_2_256_8_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 2, 8), 1, 0, 16, 380, 1,"2DDFT256_2_256_8_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_2_256_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_2_256_4_2_16bits");

end;

2ddft_27 := function()
	HDLSynthesize(stream2DDFT(256, 2, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_2_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_2_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_2_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 8, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_4_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 8, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_4_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_4_1_2_16bits");

end;

2ddft_28 := function()
	HDLSynthesize(stream2DDFT(256, 4, 16, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_8_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 16, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_8_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 32, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_16_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 32, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_16_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_16_2_2_16bits");

end;

2ddft_29 := function()
	HDLSynthesize(stream2DDFT(256, 4, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 64, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_32_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 64, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_32_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 128, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_64_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 128, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_64_4_2_16bits");

end;

2ddft_30 := function()
	HDLSynthesize(stream2DDFT(256, 4, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 256, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_128_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 256, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_128_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_128_1_2_16bits");

end;

2ddft_31 := function()
	HDLSynthesize(stream2DDFT(256, 4, 512, 1, 4), 1, 0, 16, 380, 1,"2DDFT256_4_256_4_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 512, 2, 4), 1, 0, 16, 380, 1,"2DDFT256_4_256_4_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_4_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_4_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_4_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 4, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_4_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_16_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_16_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_16_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_16_16_1_2_16bits");

end;

2ddft_32 := function()
	HDLSynthesize(stream2DDFT(256, 16, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_16_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_16_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_16_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_16_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_16_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_16_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_16_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_16_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_16_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_16_128_2_2_16bits");

end;

2ddft_33 := function()
	HDLSynthesize(stream2DDFT(256, 16, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_16_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_16_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT256_16_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT256_16_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_16_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 16, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_16_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(256, 256, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT256_256_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(256, 256, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT256_256_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 4, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_2_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 4, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_2_9_2_16bits");

end;

2ddft_34 := function()
	HDLSynthesize(stream2DDFT(512, 2, 4, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_2_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 4, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_2_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 8, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_4_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 8, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_4_9_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 8, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_4_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 8, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_4_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_4_1_2_16bits");

end;

2ddft_35 := function()
	HDLSynthesize(stream2DDFT(512, 2, 16, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_8_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 16, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_8_9_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_8_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 32, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_16_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 32, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_16_9_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_16_3_2_16bits");

end;

2ddft_36 := function()
	HDLSynthesize(stream2DDFT(512, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 64, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_32_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 64, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_32_9_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_32_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 128, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_64_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 128, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_64_9_2_16bits");

end;

2ddft_37 := function()
	HDLSynthesize(stream2DDFT(512, 2, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_64_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 256, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_128_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 256, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_128_9_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 256, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_128_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 256, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_128_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_128_1_2_16bits");

end;

2ddft_38 := function()
	HDLSynthesize(stream2DDFT(512, 2, 512, 1, 9), 1, 0, 16, 380, 1,"2DDFT512_2_256_9_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 512, 2, 9), 1, 0, 16, 380, 1,"2DDFT512_2_256_9_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 512, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_2_256_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 512, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_2_256_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_2_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_8_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_8_8_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_8_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_8_8_1_2_16bits");

end;

2ddft_39 := function()
	HDLSynthesize(stream2DDFT(512, 8, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_8_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_8_16_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_8_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_8_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_8_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_8_32_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_8_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_8_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_8_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_8_64_3_2_16bits");

end;

2ddft_40 := function()
	HDLSynthesize(stream2DDFT(512, 8, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_8_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_8_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 256, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_8_128_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 256, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_8_128_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_8_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_8_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 512, 1, 3), 1, 0, 16, 380, 1,"2DDFT512_8_256_3_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 512, 2, 3), 1, 0, 16, 380, 1,"2DDFT512_8_256_3_2_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT512_8_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(512, 8, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT512_8_256_1_2_16bits");

end;

2ddft_41 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 4, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_2_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_2_10_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_2_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_2_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_2_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_2_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_4_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_4_10_2_16bits");

end;

2ddft_42 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 8, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_4_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_4_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_8_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_8_10_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_8_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_8_5_2_16bits");

end;

2ddft_43 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_16_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_16_10_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_16_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_16_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_16_2_2_16bits");

end;

2ddft_44 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_32_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_32_10_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_32_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_32_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_32_1_2_16bits");

end;

2ddft_45 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 128, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_64_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_64_10_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_64_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_64_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_128_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_128_10_2_16bits");

end;

2ddft_46 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 256, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_128_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_128_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 1, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_256_10_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 2, 10), 1, 0, 16, 380, 1,"2DDFT1024_2_256_10_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_256_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_2_256_5_2_16bits");

end;

2ddft_47 := function()
	HDLSynthesize(stream2DDFT(1024, 2, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_2_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_2_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 8, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_4_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 8, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_4_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 16, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_8_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 16, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_8_5_2_16bits");

end;

2ddft_48 := function()
	HDLSynthesize(stream2DDFT(1024, 4, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 32, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_16_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 32, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_16_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 64, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_32_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 64, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_32_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_32_1_2_16bits");

end;

2ddft_49 := function()
	HDLSynthesize(stream2DDFT(1024, 4, 128, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_64_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 128, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_64_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 256, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_128_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 256, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_128_5_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 512, 1, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_256_5_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 512, 2, 5), 1, 0, 16, 380, 1,"2DDFT1024_4_256_5_2_16bits");

end;

2ddft_50 := function()
	HDLSynthesize(stream2DDFT(1024, 4, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 4, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_4_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_64_1_2_16bits");

end;

2ddft_51 := function()
	HDLSynthesize(stream2DDFT(1024, 32, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT1024_32_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(1024, 32, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT1024_32_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 4, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_2_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 4, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_2_11_2_16bits");

end;

2ddft_52 := function()
	HDLSynthesize(stream2DDFT(2048, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 8, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_4_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 8, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_4_11_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 16, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_8_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 16, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_8_11_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_8_1_2_16bits");

end;

2ddft_53 := function()
	HDLSynthesize(stream2DDFT(2048, 2, 32, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_16_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 32, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_16_11_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 64, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_32_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 64, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_32_11_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 128, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_64_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 128, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_64_11_2_16bits");

end;

2ddft_54 := function()
	HDLSynthesize(stream2DDFT(2048, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 256, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_128_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 256, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_128_11_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 512, 1, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_256_11_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 512, 2, 11), 1, 0, 16, 380, 1,"2DDFT2048_2_256_11_2_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(2048, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT2048_2_256_1_2_16bits");

end;

2ddft_55 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 4, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_2_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_2_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_2_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_2_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_2_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_2_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_2_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_2_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_2_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_2_2_2_16bits");

end;

2ddft_56 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_4_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_4_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_4_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_4_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_4_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_4_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_4_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_4_3_2_16bits");

end;

2ddft_57 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_8_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_8_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_8_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_8_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_8_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_8_4_2_16bits");

end;

2ddft_58 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_8_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_16_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_16_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_16_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_16_6_2_16bits");

end;

2ddft_59 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 32, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_16_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_16_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_16_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_32_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_32_12_2_16bits");

end;

2ddft_60 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 64, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_32_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_32_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_32_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_32_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_32_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_32_1_2_16bits");

end;

2ddft_61 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 128, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_64_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_64_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_64_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_64_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_64_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_64_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_64_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_64_2_2_16bits");

end;

2ddft_62 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_128_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_128_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_128_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_128_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_128_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_128_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_128_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_128_3_2_16bits");

end;

2ddft_63 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 1, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_256_12_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 2, 12), 1, 0, 16, 380, 1,"2DDFT4096_2_256_12_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_256_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_2_256_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_256_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_2_256_4_2_16bits");

end;

2ddft_64 := function()
	HDLSynthesize(stream2DDFT(4096, 2, 512, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_256_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_2_256_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_2_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_2_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_4_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_4_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_4_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_4_3_2_16bits");

end;

2ddft_65 := function()
	HDLSynthesize(stream2DDFT(4096, 4, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_8_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_8_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_8_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_8_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_8_2_2_16bits");

end;

2ddft_66 := function()
	HDLSynthesize(stream2DDFT(4096, 4, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_16_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_16_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_16_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_16_1_2_16bits");

end;

2ddft_67 := function()
	HDLSynthesize(stream2DDFT(4096, 4, 64, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_32_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_32_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_32_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_64_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_64_6_2_16bits");

end;

2ddft_68 := function()
	HDLSynthesize(stream2DDFT(4096, 4, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_64_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_128_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_128_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_128_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_128_3_2_16bits");

end;

2ddft_69 := function()
	HDLSynthesize(stream2DDFT(4096, 4, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 1, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_256_6_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 2, 6), 1, 0, 16, 380, 1,"2DDFT4096_4_256_6_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_256_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_4_256_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_4_256_2_2_16bits");

end;

2ddft_70 := function()
	HDLSynthesize(stream2DDFT(4096, 4, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 4, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_4_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 16, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_8_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 16, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_8_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 32, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_16_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 32, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_16_4_2_16bits");

end;

2ddft_71 := function()
	HDLSynthesize(stream2DDFT(4096, 8, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 64, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_32_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 64, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_32_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_32_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_32_1_2_16bits");

end;

2ddft_72 := function()
	HDLSynthesize(stream2DDFT(4096, 8, 128, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_64_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 128, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_64_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 256, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_128_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 256, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_128_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_128_2_2_16bits");

end;

2ddft_73 := function()
	HDLSynthesize(stream2DDFT(4096, 8, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 512, 1, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_256_4_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 512, 2, 4), 1, 0, 16, 380, 1,"2DDFT4096_8_256_4_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_8_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 8, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_8_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 32, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_16_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 32, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_16_3_2_16bits");

end;

2ddft_74 := function()
	HDLSynthesize(stream2DDFT(4096, 16, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 64, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_32_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 64, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_32_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 128, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_64_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 128, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_64_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_64_1_2_16bits");

end;

2ddft_75 := function()
	HDLSynthesize(stream2DDFT(4096, 16, 256, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_128_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 256, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_128_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 512, 1, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_256_3_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 512, 2, 3), 1, 0, 16, 380, 1,"2DDFT4096_16_256_3_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 16, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_16_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_64_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_64_64_2_2_16bits");

end;

2ddft_76 := function()
	HDLSynthesize(stream2DDFT(4096, 64, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_64_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_64_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_64_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_64_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_64_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_64_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT4096_64_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT4096_64_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT4096_64_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(4096, 64, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT4096_64_256_1_2_16bits");

end;

2ddft_77 := function()
	HDLSynthesize(stream2DDFT(8192, 2, 4, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_2_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 4, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_2_13_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_2_1_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 8, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_4_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 8, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_4_13_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 16, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_8_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 16, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_8_13_2_16bits");

end;

2ddft_78 := function()
	HDLSynthesize(stream2DDFT(8192, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 32, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_16_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 32, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_16_13_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 64, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_32_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 64, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_32_13_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_32_1_2_16bits");

end;

2ddft_79 := function()
	HDLSynthesize(stream2DDFT(8192, 2, 128, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_64_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 128, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_64_13_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 256, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_128_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 256, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_128_13_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 512, 1, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_256_13_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 512, 2, 13), 1, 0, 16, 380, 1,"2DDFT8192_2_256_13_2_16bits");

end;

2ddft_80 := function()
	HDLSynthesize(stream2DDFT(8192, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(8192, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT8192_2_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_2_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_2_14_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_2_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_2_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_2_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_2_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_2_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 4, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_2_1_2_16bits");

end;

2ddft_81 := function()
	HDLSynthesize(stream2DDFT(16384, 2, 8, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_4_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_4_14_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_4_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_4_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_4_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_4_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_4_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_8_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_8_14_2_16bits");

end;

2ddft_82 := function()
	HDLSynthesize(stream2DDFT(16384, 2, 16, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_8_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_8_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_8_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_8_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_16_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_16_14_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_16_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_16_7_2_16bits");

end;

2ddft_83 := function()
	HDLSynthesize(stream2DDFT(16384, 2, 32, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_16_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_16_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_32_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_32_14_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_32_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_32_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_32_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_32_2_2_16bits");

end;

2ddft_84 := function()
	HDLSynthesize(stream2DDFT(16384, 2, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_64_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_64_14_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_64_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_64_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_64_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_64_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_64_1_2_16bits");

end;

2ddft_85 := function()
	HDLSynthesize(stream2DDFT(16384, 2, 256, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_128_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_128_14_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_128_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_128_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 1, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_256_14_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 2, 14), 1, 0, 16, 380, 1,"2DDFT16384_2_256_14_2_16bits");

end;

2ddft_86 := function()
	HDLSynthesize(stream2DDFT(16384, 2, 512, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_256_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_2_256_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_2_256_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 2, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_2_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 8, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_4_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 8, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_4_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 8, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_4_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 8, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_4_1_2_16bits");

end;

2ddft_87 := function()
	HDLSynthesize(stream2DDFT(16384, 4, 16, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_8_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 16, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_8_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 16, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_8_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 16, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_8_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 32, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_16_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 32, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_16_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 32, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_16_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 32, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_16_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 64, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_32_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 64, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_32_7_2_16bits");

end;

2ddft_88 := function()
	HDLSynthesize(stream2DDFT(16384, 4, 64, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_32_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 64, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_32_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 128, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_64_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 128, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_64_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 128, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_64_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 128, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_64_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 256, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_128_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 256, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_128_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_128_1_2_16bits");

end;

2ddft_89 := function()
	HDLSynthesize(stream2DDFT(16384, 4, 512, 1, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_256_7_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 512, 2, 7), 1, 0, 16, 380, 1,"2DDFT16384_4_256_7_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 4, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_4_256_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 256, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_128_128_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 256, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_128_128_2_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 256, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_128_128_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 256, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_128_128_1_2_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 512, 1, 2), 1, 0, 16, 380, 1,"2DDFT16384_128_256_2_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 512, 2, 2), 1, 0, 16, 380, 1,"2DDFT16384_128_256_2_2_16bits");

end;

2ddft_90 := function()
	HDLSynthesize(stream2DDFT(16384, 128, 512, 1, 1), 1, 0, 16, 380, 1,"2DDFT16384_128_256_1_1_16bits");
	HDLSynthesize(stream2DDFT(16384, 128, 512, 2, 1), 1, 0, 16, 380, 1,"2DDFT16384_128_256_1_2_16bits");

end;
