--  script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student.
DELIMITER $$
CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN new_user_id INT)
BEGIN
  UPDATE users
  SET average_score = (
    SELECT SUM(weight * score) / SUM(weight)
    FROM corrections, projects
    WHERE corrections.user_id = new_user_id
    AND corrections.project_id = projects.id)
    WHERE users.id = new_user_id;
END $$
DELIMITER ;
