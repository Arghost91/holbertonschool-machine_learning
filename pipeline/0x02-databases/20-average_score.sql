--  script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser  (
  IN user_id INT)
BEGIN
  UPDATE users
  SET average_score=(
    SELECT AVG(score)
    FROM corrections
    WHERE corrections.user_id = user_id)
  WHERE id=user_id;
END $$
DELIMITER ;
