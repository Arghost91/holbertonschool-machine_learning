--  script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student.
DELIMITER $$
CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN new_user_id INT)
BEGIN
  UPDATE users
  SET average_score = (
    SELECT SUM(weight * score) / SUM(weight)
    FROM corrections, projects
    WHERE corrections.user_id = user_id
    AND corrections.project_id = projects_id)
    WHERE id = user_id;
END $$
DELIMITER ;
