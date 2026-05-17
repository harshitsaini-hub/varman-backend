class Solution {
    public int theMaximumAchievableX(int num, int t) {
        if (num == 0 || t == 0) return 0;
        int sol = 0;
        sol = num + 2*t;
        return sol;
    }
}
